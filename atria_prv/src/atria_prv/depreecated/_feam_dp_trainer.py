from __future__ import annotations

import copy
from collections import OrderedDict
from math import sqrt

import torch
from atria_logger._api import get_logger
from atria_ml.data_pipeline._data_pipeline import DataPipeline

from atria_datasets.core.dataset._partitioning import DatasetPartitioner
from atria_prv.trainers._feam_dp_client import FeAmDPClient
from atria_prv.trainers._fl_client import FLClientOutput
from atria_prv.trainers._fl_trainer import FLTrainer, FLTrainerState
from atria_prv.configs import FeAmDPTrainingTaskConfig

logger = get_logger(__name__)


class FeAmDPTrainer(FLTrainer):
    """Only overrides _build() (DP-aware clients + sigma computation + server Adam
    optimizer) and _run_update() (Adam-based aggregation instead of plain averaging).
    train(), test(), _build_test_engine(), _build_global_tracking_engine() are all
    inherited UNCHANGED from FLTrainer -- the payoff of FLClient._build_train_engine
    being the documented override seam: the whole round loop / checkpointing / resume /
    tracking-validation machinery is reused for free."""

    def __init__(self, config: FeAmDPTrainingTaskConfig) -> None:
        self._config = config
        self._state: FLTrainerState = self._build()

    def _build(self) -> FLTrainerState:
        self._initialize_runtime()

        tb_logger = self._setup_logging()

        train_transform = self._config.model_pipeline.train_transform
        eval_transform = self._config.model_pipeline.eval_transform

        dataset = self._config.data.build_dataset()
        dataset.apply_transforms(
            train_transform=train_transform, eval_transform=eval_transform
        )
        labels = dataset.metadata.dataset_labels
        logger.info(f"Dataset:\n{dataset}")

        global_model_pipeline = self._config.model_pipeline.build(labels=labels)
        logger.info(global_model_pipeline.ops.summarize())

        logger.info("Data transforms:")
        logger.info(f"Train transform:\n{train_transform}")
        logger.info(f"Eval transform:\n{eval_transform}")

        global_data_pipeline = DataPipeline(dataset=dataset)

        # privacy setup -- computed once, before clients are built
        from opacus.accountants.utils import create_accountant, get_noise_multiplier

        dp_config = self._config.dp_config
        fl_config = self._config.fl_config
        target_delta = dp_config.target_delta or (1.0 / len(dataset.train))
        # All clients participate every round, so client_fraction plays no role in
        # privacy: the DP subsampling rate is dp_config.sample_rate (applied per client
        # at the data level). Each client takes a single local DP-SGD step per round, so
        # total_steps is calculated over the full run as steps_per_round * num_rounds.
        steps_per_round = 1
        total_steps = steps_per_round * fl_config.num_rounds
        sigma = get_noise_multiplier(
            target_epsilon=dp_config.target_epsilon,
            target_delta=target_delta,
            sample_rate=dp_config.sample_rate,
            steps=total_steps,
            accountant=dp_config.accountant,
        )
        self._sigma = sigma
        self._sigma_k = sigma / sqrt(fl_config.total_num_clients)
        self._target_delta = target_delta
        self._q = dp_config.sample_rate
        self._privacy_accountant = create_accountant(mechanism=dp_config.accountant)
        logger.info(
            f"FeAm-DP privacy setup: target_epsilon={dp_config.target_epsilon}, "
            f"target_delta={target_delta}, sample_rate(q)={self._q}, "
            f"steps_per_round={steps_per_round}, num_rounds={fl_config.num_rounds}, "
            f"total_steps={total_steps} "
            f"-> sigma={sigma:.4f}, sigma_k={self._sigma_k:.4f} (K={fl_config.total_num_clients})"
        )

        num_clients = fl_config.total_num_clients
        partitioner = DatasetPartitioner(
            dataset=dataset, n_partitions=num_clients, seed=fl_config.seed
        )
        logger.info(f"Partition sizes per client: {partitioner.partition_sizes()}")

        clients: list[FeAmDPClient] = []
        for client_id in range(num_clients):
            partition = partitioner.get_partition(client_id)
            client_dataset = copy.copy(dataset)
            client_dataset._split_iterators = dict(partition.split_iterators)
            clients.append(
                FeAmDPClient(
                    client_id=client_id,
                    config=self._config,
                    data_pipeline=DataPipeline(dataset=client_dataset),
                    labels=labels,
                    device=torch.device(self._device),
                    tb_logger=tb_logger,
                    dp_config=dp_config,
                    sigma_k=self._sigma_k,
                )
            )

        # built ONCE, persists across rounds -- this IS the algorithm's m_t/v_t state,
        # via native PyTorch Adam instead of hand-rolled bias-corrected updates.
        server_optim = self._config.server_optim
        self._server_optimizer = torch.optim.Adam(
            global_model_pipeline._model.parameters(),
            lr=server_optim.lr,
            betas=(server_optim.beta1, server_optim.beta2),
            eps=server_optim.eps,
        )

        return FLTrainerState(
            dataset=dataset,
            server_model_pipeline=global_model_pipeline,
            server_data_pipeline=global_data_pipeline,
            clients=clients,
            tb_logger=tb_logger,
        )

    def _run_update(self, selected_client_ids: list[int]) -> None:
        """Same accumulation-loop shape as FLTrainer._run_update (plain average over
        len(selected_client_ids), not weighted by n_k/n -- matches FLTrainer's own
        established "minimal, unweighted" convention, and DatasetPartitioner partitions
        are equal-sized anyway) -- but client_update() returns pseudo-gradients
        (named_parameters()-keyed) here, and finalization applies a server Adam step
        instead of load_state_dict."""
        global_params = OrderedDict(
            (k, v.detach().cpu())
            for k, v in self._state.server_model_pipeline._model.state_dict().items()
        )

        accumulated: OrderedDict[str, torch.Tensor] | None = None
        for client_id in selected_client_ids:
            logger.info(f"[Client {client_id}] update starting")
            output: FLClientOutput = self._state.clients[client_id].client_update(
                global_params
            )
            if accumulated is None:
                accumulated = output.params
            else:
                for k, v in output.params.items():
                    accumulated[k] += v
            logger.info(
                f"[Client {client_id}] update finished; metrics: {output.metrics}"
            )
            del output

        averaged_grad = {
            k: v / len(selected_client_ids) for k, v in accumulated.items()
        }

        self._server_optimizer.zero_grad()
        for name, param in self._state.server_model_pipeline._model.named_parameters():
            if name in averaged_grad:
                param.grad = averaged_grad[name].to(param.device)
        self._server_optimizer.step()
        logger.info(
            f"Applied server Adam step from {len(selected_client_ids)} client pseudo-gradient(s)"
        )

        self._privacy_accountant.step(noise_multiplier=self._sigma, sample_rate=self._q)
        epsilon = self._privacy_accountant.get_epsilon(delta=self._target_delta)
        logger.info(
            f"Privacy spent so far: epsilon={epsilon:.3f} "
            f"(target={self._config.dp_config.target_epsilon}), delta={self._target_delta}"
        )
