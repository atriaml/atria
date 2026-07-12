from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger._api import get_logger
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._trainer import Trainer
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

from atria_prv.dp._dp_trainer_engine import (
    DPTrainerEngine,
    DPTrainerEngineConfig,
    DPTrainerEngineDependencies,
)
from atria_prv.dp.configs import DPTrainingTaskConfig
from atria_prv.dp.opacus.privacy_engine import _PrivacyEngine

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclass
class DPTrainerState:
    data_pipeline: DataPipeline
    model_pipeline: ModelPipeline
    privacy_engine: _PrivacyEngine
    tb_logger: TensorboardLogger | None = None

    @property
    def dataset(self) -> Dataset:
        return self.data_pipeline.dataset


class DPTrainer(Trainer):
    def __init__(self, config: DPTrainingTaskConfig) -> None:
        self._config = config
        self._state: DPTrainerState = self._build()

    def _build_train_engine(self) -> DPTrainerEngine:
        import torch

        train_dataloader = self._state.data_pipeline.train_dataloader(
            batch_size=self._config.data.train_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )

        return DPTrainerEngine(
            config=DPTrainerEngineConfig(
                max_epochs=self._config.trainer.max_epochs,
                outputs_to_running_avg=self._config.trainer.outputs_to_running_avg,
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                clear_cuda_cache=self._config.trainer.clear_cuda_cache,
                stop_on_nan=False,
                eval_training=self._config.trainer.eval_training,
                validate_every_n_epochs=self._config.trainer.validate_every_n_epochs,
                visualize_every_n_epochs=self._config.trainer.visualize_every_n_epochs,
                optimizer=self._config.trainer.optimizer,
                lr_scheduler=self._config.trainer.lr_scheduler,
                model_ema=self._config.trainer.model_ema,
                warmup=self._config.trainer.warmup,
                model_checkpoint=self._config.trainer.model_checkpoint,
                gradient=self._config.trainer.gradient,
                dp_config=self._config.dp_config,
            ),
            deps=DPTrainerEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=train_dataloader,
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
                run_config=self._config,
                privacy_engine=self._state.privacy_engine,
            ),
        )

    def _build(self) -> DPTrainerState:
        self._initialize_runtime()

        # setup logging
        tb_logger = self._setup_logging()

        # get model transforms
        train_transform = self._config.model_pipeline.train_transform
        eval_transform = self._config.model_pipeline.eval_transform

        # build dataset
        dataset = self._config.data.build_dataset()

        # apply transforms
        dataset.apply_transforms(
            train_transform=train_transform, eval_transform=eval_transform
        )

        # load labels
        labels = dataset.metadata.dataset_labels

        # log dataset info
        logger.info(f"Dataset:\n{dataset}")

        # build model pipeline
        model_pipeline = self._config.model_pipeline.build(labels=labels)

        # call model validator -- privacy updates
        from opacus.validators import ModuleValidator

        if not ModuleValidator.is_valid(model_pipeline._model):
            model_pipeline._model = ModuleValidator.fix(model_pipeline._model)

        # setup privacy engine -- privacy updates
        privacy_engine = _PrivacyEngine(accountant=self._config.dp_config.accountant)

        # log model pipeline
        logger.info(model_pipeline.ops.summarize())

        logger.info("Data transforms:")
        logger.info(f"Train transform:\n{train_transform}")
        logger.info(f"Eval transform:\n{eval_transform}")

        # build data pipeline
        data_pipeline = DataPipeline(dataset=dataset)

        return DPTrainerState(
            data_pipeline=data_pipeline,
            model_pipeline=model_pipeline,
            privacy_engine=privacy_engine,
            tb_logger=tb_logger,
        )
