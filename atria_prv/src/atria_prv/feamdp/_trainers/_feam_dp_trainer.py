from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from atria_datasets.core.dataset._partitioning import DatasetPartitioner
from atria_logger._api import get_logger
from atria_ml.task_pipelines._trainer import Trainer, TrainerState
from opacus.accountants.utils import create_accountant, get_noise_multiplier

from atria_prv.dp.configs import DPConfig
from atria_prv.feamdp._feam_aggregation import FeAmAggregation
from atria_prv.feamdp.configs import (
    FeAmDPClientTrainingTaskConfig,
    FeAmDPTrainingTaskConfig,
)
from atria_prv.fl._engines._fl_training_engine import (
    FLTrainingEngine,
    FLTrainingEngineConfig,
    FLTrainingEngineDependencies,
)
from atria_prv.fl._engines._fl_validation_engine import (
    FLValidationEngine,
    FLValidationEngineConfig,
    FLValidationEngineDependencies,
)
from atria_prv.fl.configs import FLClientDataConfig

logger = get_logger(__name__)


@dataclass
class FeAmDPTrainerState(TrainerState):
    client_training_task_configs: list[FeAmDPClientTrainingTaskConfig] | None = None


class FeAmDPTrainer(Trainer):
    _config: FeAmDPTrainingTaskConfig

    def _build(self) -> FeAmDPTrainerState:
        state = super()._build()

        # prepare privacy args for clients
        target_delta = 1.0 / len(state.data_pipeline.dataset.train)

        # make train dataloader for full dataset to compute global sample rate
        train_dataloader = state.data_pipeline.train_dataloader(
            batch_size=self._config.data.train_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )

        # this is the sample rate computed from global dataloader. Since we want same rate between DP and FeAm-DP
        sample_rate = 1.0 / len(train_dataloader)

        # base self._config.fl_config.num_rounds matches the total epochs in DP
        # dividng by sample rate equals total optimziation steps in dp to total optimization steps in feAm-DP
        updated_num_rounds = self._config.fl_config.num_rounds / sample_rate

        # update the config
        self._config = self._config.model_copy(
            update={
                "dp_config": self._config.dp_config.model_copy(
                    update={"sample_rate": sample_rate, "target_delta": target_delta}
                ),
                "fl_config": self._config.fl_config.model_copy(
                    update={"num_rounds": updated_num_rounds}
                ),
            }
        )

        noise_multiplier = get_noise_multiplier(
            target_epsilon=self._config.dp_config.target_epsilon,
            target_delta=self._config.dp_config.target_delta,
            sample_rate=self._config.dp_config.sample_rate,
            steps=self._config.fl_config.num_rounds,
            accountant=self._config.dp_config.accountant,
        )
        client_noise_multiplier = noise_multiplier / sqrt(
            self._config.fl_config.total_num_clients
        )
        self._privacy_accountant = create_accountant(
            mechanism=self._config.dp_config.accountant
        )

        trainable_parameters = state.model_pipeline.trainable_parameters
        assert (
            "default" in trainable_parameters and len(trainable_parameters.keys()) == 1
        ), "Only single optimizer supported."
        self._aggregation = FeAmAggregation(
            optimizer=self._config.global_optimizer.build(
                parameters=trainable_parameters["default"]
            ),
            privacy_accountant=self._privacy_accountant,
            sample_rate=self._config.dp_config.sample_rate,
            noise_multiplier=noise_multiplier,
            target_delta=self._config.dp_config.target_delta,
        )
        logger.info(
            f"FeAm-DP privacy setup: target_epsilon={self._config.dp_config.target_epsilon}, "
            f"target_delta={target_delta}, sample_rate(q)={self._config.dp_config.sample_rate}, "
            f"num_rounds={self._config.fl_config.num_rounds}, "
            f"-> sigma={noise_multiplier:.4f}, sigma_k={client_noise_multiplier:.4f} (K={self._config.fl_config.total_num_clients})"
        )

        num_clients = self._config.fl_config.total_num_clients
        partitioner = DatasetPartitioner(
            dataset=state.data_pipeline.dataset,
            n_partitions=num_clients,
            seed=self._config.fl_config.seed,
        )
        partitioner.compute()
        logger.info(f"Partition sizes per client: {partitioner.partition_sizes()}")

        client_training_task_configs: list[FeAmDPClientTrainingTaskConfig] = []
        for client_id in range(num_clients):
            client_data_config = FLClientDataConfig.from_data_config(
                self._config.data,
                partition_id=client_id,
                partition_cache_dir=partitioner.partition_cache_dir,
            )
            client_training_task_config = FeAmDPClientTrainingTaskConfig(
                model_pipeline=self._config.model_pipeline,
                data=client_data_config,
                dp_config=DPConfig(
                    noise_multiplier=client_noise_multiplier,
                    target_epsilon=None,
                    target_delta=self._config.dp_config.target_delta,
                    max_grad_norm=self._config.dp_config.max_grad_norm,
                    sample_rate=self._config.dp_config.sample_rate,
                    clipping=self._config.dp_config.clipping,
                    accountant=self._config.dp_config.accountant,
                    max_physical_batch_size=self._config.dp_config.max_physical_batch_size,
                ),
                client_id=client_id,
                partition_cache_dir=partitioner.partition_cache_dir,
            )
            client_training_task_configs.append(client_training_task_config)

        return FeAmDPTrainerState(
            model_pipeline=state.model_pipeline,
            data_pipeline=state.data_pipeline,
            tb_logger=state.tb_logger,
            client_training_task_configs=client_training_task_configs,
        )

    def _build_train_engine(self) -> FLTrainingEngine:
        return FLTrainingEngine(
            config=FLTrainingEngineConfig(
                max_epochs=self._config.fl_config.num_rounds,
                total_num_clients=self._config.fl_config.total_num_clients,
                client_fraction=self._config.fl_config.client_fraction,
                seed=self._config.fl_config.seed,
                logging=self._config.logging,
                model_checkpoint=self._config.model_checkpoint,
                outputs_to_running_avg=[],
            ),
            deps=FLTrainingEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                # one round == one epoch, driven by a trivial single-item loader
                dataloader=torch.utils.data.DataLoader([0], batch_size=1),
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
                run_config=self._config,
                client_training_task_configs=self._state.client_training_task_configs,
                aggregation_strategy=self._aggregation,
            ),
        )

    def _build_visualization_engine(self, trainer_engine: FLTrainingEngine):
        return None

    def _build_validation_engine(
        self, trainer_engine: FLTrainingEngine
    ) -> FLValidationEngine:
        import torch

        validation_dataloader = self._state.data_pipeline.validation_dataloader(
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        return FLValidationEngine(
            config=FLValidationEngineConfig(
                logging=self._config.logging,
                run_every_n_epochs=self._config.validate_every_n_rounds,
                run_on_start=True,
                model_checkpoint=self._config.model_checkpoint,
            ),
            deps=FLValidationEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=validation_dataloader,
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
                training_engine=trainer_engine,
            ),
        )

    def train(self) -> None:
        train_engine = self._build_train_engine()
        if self._config.do_validation:
            self._build_validation_engine(trainer_engine=train_engine)

        # save the run configuration used for training
        self._config.save_to_json()  # type: ignore
        train_engine.run()
