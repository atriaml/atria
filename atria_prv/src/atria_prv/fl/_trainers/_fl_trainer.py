from __future__ import annotations

from dataclasses import dataclass

import torch
from atria_datasets.core.dataset._partitioning import DatasetPartitioner
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger._api import get_logger
from atria_ml.task_pipelines._trainer import Trainer, TrainerState

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
from atria_prv.fl.configs import (
    FLClientDataConfig,
    FLClientTrainingTaskConfig,
    FLTrainingTaskConfig,
)

logger = get_logger(__name__)


@dataclass
class FLTrainerState(TrainerState):
    client_training_task_configs: list[FLClientTrainingTaskConfig] | None = None


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

        client_training_task_configs: list[FLClientTrainingTaskConfig] = []
        for client_id in range(num_clients):
            client_data_config = FLClientDataConfig.from_data_config(
                self._config.data,
                partition_id=client_id,
                partition_cache_dir=partitioner.partition_cache_dir,
            )
            client_training_task_config = (
                FLClientTrainingTaskConfig.from_training_task_config(
                    training_task_config=self._config,
                    data_config=client_data_config,
                    client_id=client_id,
                    partition_cache_dir=partitioner.partition_cache_dir,
                )
            )
            client_training_task_configs.append(client_training_task_config)

        return FLTrainerState(
            model_pipeline=state.model_pipeline,
            data_pipeline=state.data_pipeline,
            tb_logger=state.tb_logger,
            client_training_task_configs=client_training_task_configs,
        )

    def _build_train_engine(self) -> FLTrainingEngine:
        fl_config = self._config.fl_config
        return FLTrainingEngine(
            config=FLTrainingEngineConfig(
                max_epochs=fl_config.num_rounds,
                total_num_clients=fl_config.total_num_clients,
                client_fraction=fl_config.client_fraction,
                seed=fl_config.seed,
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                outputs_to_running_avg=[],
                model_checkpoint=self._config.trainer.model_checkpoint,
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
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                run_every_n_epochs=self._config.trainer.validate_every_n_epochs,
                run_on_start=True,
                use_ema=self._config.use_ema_for_evaluation,
                early_stopping=self._config.trainer.early_stopping,
                model_checkpoint=self._config.trainer.model_checkpoint,
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
