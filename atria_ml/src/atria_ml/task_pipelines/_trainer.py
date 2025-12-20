from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger import enable_file_logging
from atria_logger._api import get_logger
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from omegaconf import OmegaConf

from atria_ml.configs._base import RunConfig
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._utilities import _get_env_info, _initialize_torch
from atria_ml.training.engines._test_engine import (
    NoCheckpointFoundError,
    TestEngine,
    TestEngineConfig,
    TestEngineDependencies,
)
from atria_ml.training.engines._trainer import (
    TrainerEngine,
    TrainerEngineConfig,
    TrainerEngineDependencies,
)
from atria_ml.training.engines._validation_engine import (
    ValidationEngine,
    ValidationEngineConfig,
    ValidationEngineDependencies,
)
from atria_ml.training.engines._visualization_engine import (
    VisualizationEngine,
    VisualizationEngineConfig,
    VisualizationEngineDependencies,
)
from atria_ml.training.engines.utilities import _format_metrics_for_logging

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclass
class TrainerState:
    data_pipeline: DataPipeline
    model_pipeline: ModelPipeline
    tb_logger: TensorboardLogger | None = None

    @property
    def dataset(self) -> Dataset:
        return self.data_pipeline.dataset


class Trainer:
    def __init__(self, config: RunConfig, local_rank: int = 0) -> None:
        self._config = config
        self._state: TrainerState = self._build(local_rank=local_rank)

    def _initialize_runtime(self, local_rank: int) -> None:
        # Log system information
        env_info = _get_env_info()

        # initialize training
        _initialize_torch(
            seed=self._config.env.seed, deterministic=self._config.env.deterministic
        )

        # initialize torch device (cpu or gpu)
        self._device = local_rank

        # log env info and run configuration
        logger.info(
            f"Environment info:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(env_info)), indent=4)}"
        )
        logger.info(
            f"Run configuration:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(self._config.model_dump())), indent=4)}"
        )
        logger.info(f"Seed set to {self._config.env.seed} on device: {self._device}")

    def _setup_logging(self) -> TensorboardLogger | None:
        import ignite.distributed as idist
        from ignite.handlers import TensorboardLogger

        if idist.get_rank() == 0:
            log_dir = Path(self._config.env.run_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TensorboardLogger(log_dir=log_dir)

            enable_file_logging(str(Path(self._config.env.run_dir) / "training.log"))
        else:
            tb_logger = None
        return tb_logger

    def _build_train_engine(self) -> TrainerEngine:
        import torch

        train_dataloader = self._state.data_pipeline.train_dataloader(
            batch_size=self._config.data.train_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        return TrainerEngine(
            config=TrainerEngineConfig(
                max_epochs=self._config.trainer.max_epochs,
                outputs_to_running_avg=self._config.trainer.outputs_to_running_avg,
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                clear_cuda_cache=self._config.trainer.clear_cuda_cache,
                stop_on_nan=self._config.trainer.stop_on_nan,
                eval_training=self._config.trainer.eval_training,
                validate_every_n_epochs=self._config.trainer.validate_every_n_epochs,
                visualize_every_n_epochs=self._config.trainer.visualize_every_n_epochs,
                optimizer=self._config.trainer.optimizer,
                lr_scheduler=self._config.trainer.lr_scheduler,
                model_ema=self._config.trainer.model_ema,
                warmup=self._config.trainer.warmup,
                model_checkpoint=self._config.trainer.model_checkpoint,
                gradient=self._config.trainer.gradient,
            ),
            deps=TrainerEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=train_dataloader,
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
                run_config=self._config,
            ),
        )

    def _build_validation_engine(
        self, trainer_engine: TrainerEngine
    ) -> ValidationEngine:
        import torch

        validation_dataloader = self._state.data_pipeline.validation_dataloader(
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        return ValidationEngine(
            config=ValidationEngineConfig(
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
            deps=ValidationEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=validation_dataloader,
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
                training_engine=trainer_engine,
            ),
        )

    def _build_visualization_engine(
        self, trainer_engine: TrainerEngine
    ) -> VisualizationEngine:
        import torch

        # visualization uses the same dataloader as validation
        validation_dataloader = self._state.data_pipeline.validation_dataloader(
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        return VisualizationEngine(
            config=VisualizationEngineConfig(
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                run_every_n_epochs=self._config.trainer.validate_every_n_epochs,
                run_on_start=True,
                use_ema=self._config.use_ema_for_evaluation,
            ),
            deps=VisualizationEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=validation_dataloader,
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
                training_engine=trainer_engine,
            ),
        )

    def _build_test_engine(self) -> TestEngine:
        import torch

        test_dataloader = self._state.data_pipeline.test_dataloader(
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        return TestEngine(
            config=TestEngineConfig(
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                save_model_outputs_to_disk=self._config.save_test_outputs_to_disk,
            ),
            deps=TestEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=test_dataloader,
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
            ),
        )

    def _build(self, local_rank: int) -> TrainerState:
        self._initialize_runtime(local_rank=local_rank)

        # setup logging
        tb_logger = self._setup_logging()

        # build dataset
        dataset = self._config.data.build_dataset()

        # load labels
        labels = dataset.metadata.dataset_labels

        # log dataset info
        logger.info(f"Dataset:\n{dataset}")

        # build model pipeline
        model_pipeline = self._config.model_pipeline.build(labels=labels)

        # log model pipeline
        logger.info(model_pipeline.ops.summarize())

        # get model transforms
        train_transform = model_pipeline.config.train_transform
        eval_transform = model_pipeline.config.eval_transform
        if dataset.train is not None:
            dataset.train.output_transform = train_transform
        if dataset.validation is not None:
            dataset.validation.output_transform = eval_transform
        if dataset.test is not None:
            dataset.test.output_transform = eval_transform

        logger.info("Data transforms:")
        logger.info(f"Train transform:\n{train_transform}")
        logger.info(f"Eval transform:\n{eval_transform}")

        # build data pipeline
        data_pipeline = DataPipeline(dataset=dataset)

        return TrainerState(
            data_pipeline=data_pipeline,
            model_pipeline=model_pipeline,
            tb_logger=tb_logger,
        )

    def train(self) -> None:
        train_engine = self._build_train_engine()
        if self._config.do_validation:
            self._build_validation_engine(trainer_engine=train_engine)
        if self._config.do_visualization:
            self._build_visualization_engine(trainer_engine=train_engine)

        # save the run configuration used for training
        self._config.save_to_json()  # type: ignore
        train_engine.run(checkpoint_path=self._config.trainer.resume_checkpoint_path)

    def test(self):
        test_engine = self._build_test_engine()

        if self._config.metrics_file_exists() and not self._config.reevaluate_metrics:
            logger.warning(
                f"Test metrics file {self._config.get_metrics_file_path()} already exists. Skipping testing step."
            )
            return

        metrics = {}
        for checkpoint_type in ["best", "last"]:
            try:
                state = test_engine.run_with_checkpoint_type(
                    checkpoint_type=checkpoint_type
                )
                if state is not None:
                    metrics[checkpoint_type] = state.metrics
            except NoCheckpointFoundError:
                pass  # ignore if no checkpoint found

        metrics = _format_metrics_for_logging(metrics)
        logger.info("Test metrics:")
        logger.info(json.dumps(metrics, indent=4))

        # serialize test metrics
        self._config.dump_metrics_file(data=metrics)  #  type: ignore
        return metrics

    def run(self) -> None:
        if self._config.do_train:
            self.train()
        if self._config.do_test:
            self.test()
