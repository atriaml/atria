from __future__ import annotations

from dataclasses import dataclass

import torch
from atria_datasets.core.dataset._partitioning import DatasetPartitioner
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger._api import get_logger
from atria_ml.task_pipelines._trainer import Trainer, TrainerState

from atria_prv.configs import ClientDataConfig, FLTrainingTaskConfig
from atria_prv.engines._fl_trainer_engine import (
    FLEngine,
    FLEngineConfig,
    FLEngineDependencies,
)
from atria_prv.trainers._fl_client_trainer import FLClientTrainer

logger = get_logger(__name__)


@dataclass
class FLTrainerState(TrainerState):
    clients: list[FLClientTrainer] | None = None


class FLTrainer(Trainer):
    def __init__(self, config: FLTrainingTaskConfig) -> None:
        self._config = config
        self._state: FLTrainerState = self._build()

    def _build(self) -> FLTrainerState:
        state = super()._build()

        num_clients = self._config.fl_config.total_num_clients
        partitioner = DatasetPartitioner(
            dataset=state.data_pipeline.dataset,
            n_partitions=num_clients,
            seed=self._config.fl_config.seed,
        )
        partitioner.compute()
        logger.info(f"Partition sizes per client: {partitioner.partition_sizes()}")

        client_trainers: list[FLClientTrainer] = []
        for client_id in range(num_clients):
            client_data_config = ClientDataConfig.from_data_config(
                self._config.data,
                partition_id=client_id,
                partition_cache_dir=partitioner.partition_cache_dir,
            )
            client_training_task_config = self._config.model_copy(
                update={"data": client_data_config}
            )
            client_trainers.append(
                FLClientTrainer(
                    client_id=client_id,
                    partition_cache_dir=partitioner.partition_cache_dir,
                    config=client_training_task_config,
                )
            )

        return FLTrainerState(
            model_pipeline=state.model_pipeline,
            data_pipeline=state.data_pipeline,
            tb_logger=state.tb_logger,
            clients=client_trainers,
        )

    def _build_train_engine(self) -> FLEngine:
        fl_config = self._config.fl_config
        return FLEngine(
            config=FLEngineConfig(
                max_epochs=fl_config.num_rounds,
                total_num_clients=fl_config.total_num_clients,
                client_fraction=fl_config.client_fraction,
                seed=fl_config.seed,
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                outputs_to_running_avg=[],
                eval_training=False,
                stop_on_nan=False,
                clear_cuda_cache=False,
                validate_every_n_epochs=self._config.trainer.validate_every_n_epochs,
                model_checkpoint=self._config.trainer.model_checkpoint,
            ),
            deps=FLEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                # one round == one epoch, driven by a trivial single-item loader
                dataloader=torch.utils.data.DataLoader([0], batch_size=1),
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
                run_config=self._config,
                client_trainers=self._state.clients,
            ),
        )

    def _build_visualization_engine(self, trainer_engine: FLEngine):
        # Base Trainer.train() only calls this when do_visualization is set; FL is a
        # global-tracking-only loop, so skip it gracefully instead of building one.
        logger.warning(
            "do_visualization is not supported by FLTrainer (global-tracking-only "
            "federated loop); skipping visualization engine."
        )
        return None
