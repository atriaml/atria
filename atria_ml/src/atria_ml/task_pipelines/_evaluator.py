from __future__ import annotations

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

from atria_ml.configs._task import EvaluationTaskConfig
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._utilities import _get_env_info, _initialize_torch
from atria_ml.training.engines._test_engine import (
    TestEngine,
    TestEngineConfig,
    TestEngineDependencies,
)
from atria_ml.training.engines.utilities import _format_metrics_for_logging

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclass
class EvaluatorState:
    data_pipeline: DataPipeline
    model_pipeline: ModelPipeline
    tb_logger: TensorboardLogger | None = None

    @property
    def dataset(self) -> Dataset:
        return self.data_pipeline.dataset


class Evaluator:
    def __init__(self, config: EvaluationTaskConfig, local_rank: int = 0) -> None:
        self._config = config
        self._state: EvaluatorState = self._build(local_rank=local_rank)

    def _initialize_runtime(self, local_rank: int) -> None:
        import ignite.distributed as idist
        import torch

        # Log system information
        env_info = _get_env_info()

        # initialize training
        _initialize_torch(
            seed=self._config.env.seed, deterministic=self._config.env.deterministic
        )

        # initialize torch device (cpu or gpu)
        if torch.cuda.is_available():
            self._device = idist.device()
        else:
            self._device = "cpu"

        # log env info and run configuration
        logger.info(
            f"Environment info:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(env_info)), indent=4)}"
        )
        dumped = self._config.model_dump()
        logger.info(
            f"Run configuration:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(dumped)), indent=4)}"
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

    def _build(self, local_rank: int) -> EvaluatorState:
        self._initialize_runtime(local_rank=local_rank)

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

        # log model pipeline
        logger.info(model_pipeline.ops.summarize())

        logger.info("Data transforms:")
        logger.info(f"Train transform:\n{train_transform}")
        logger.info(f"Eval transform:\n{eval_transform}")

        # build data pipeline
        data_pipeline = DataPipeline(dataset=dataset)

        return EvaluatorState(
            data_pipeline=data_pipeline,
            model_pipeline=model_pipeline,
            tb_logger=tb_logger,
        )

    def run(self) -> dict:
        test_engine = self._build_test_engine()
        state = test_engine.run(checkpoint_path=self._config.eval_checkpoint)
        metrics = _format_metrics_for_logging(state.metrics)
        logger.info("Test metrics:")
        logger.info(metrics)

        return metrics
