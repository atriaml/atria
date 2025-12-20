from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger._api import get_logger
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._utilities import _get_env_info, _initialize_torch
from atria_ml.training.engines.utilities import _format_metrics_for_logging
from omegaconf import OmegaConf

from atria_insights.core.configs.explainer_config import ExplainerRunConfig
from atria_insights.core.engines._explanation_engine import (
    ExplanationEngine,
    ExplanationEngineConfig,
    ExplanationEngineDependencies,
)
from atria_insights.core.model_pipelines._model_pipeline import ExplainableModelPipeline

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclass
class ModelExplainerState:
    data_pipeline: DataPipeline
    x_model_pipeline: ExplainableModelPipeline
    tb_logger: TensorboardLogger | None = None

    @property
    def dataset(self) -> Dataset:
        return self.data_pipeline.dataset


class ModelExplainer:
    def __init__(self, config: ExplainerRunConfig, local_rank: int = 0) -> None:
        self._config = config
        self._state: ModelExplainerState = self._build(local_rank=local_rank)

    def _initialize_runtime(self, local_rank: int) -> None:
        # Log system information
        env_info = _get_env_info()

        # initialize training
        _initialize_torch(
            seed=self._config.env_config.seed,
            deterministic=self._config.env_config.deterministic,
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
        logger.info(
            f"Seed set to {self._config.env_config.seed} on device: {self._device}"
        )

    def _setup_logging(self) -> TensorboardLogger | None:
        import ignite.distributed as idist
        from ignite.handlers import TensorboardLogger

        if idist.get_rank() == 0:
            log_dir = Path(self._config.env_config.run_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TensorboardLogger(log_dir=log_dir)
        else:
            tb_logger = None
        return tb_logger

    def _build_explanation_engine(self) -> ExplanationEngine:
        import torch

        test_dataloader = self._state.data_pipeline.test_dataloader(
            batch_size=self._config.data_config.eval_batch_size,
            num_workers=self._config.data_config.num_workers,
            pin_memory=self._config.data_config.pin_memory,
        )
        return ExplanationEngine(
            config=ExplanationEngineConfig(
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
            ),
            deps=ExplanationEngineDependencies(
                model_pipeline=self._state.x_model_pipeline._model_pipeline,
                x_model_pipeline=self._state.x_model_pipeline,
                dataloader=test_dataloader,
                device=torch.device(self._device),
                output_dir=self._config.env_config.run_dir,
                tb_logger=self._state.tb_logger,
            ),
        )

    def _build(self, local_rank: int = 0) -> ModelExplainerState:
        self._initialize_runtime(local_rank=local_rank)

        # setup logging
        tb_logger = self._setup_logging()

        # build dataset
        dataset = self._config.data_config.build_dataset()

        # load labels
        labels = dataset.metadata.dataset_labels

        # log dataset info
        logger.info(f"Dataset:\n{dataset}")

        # build model pipeline
        x_model_pipeline = self._config.x_model_pipeline_config.build(labels=labels)

        # log model pipeline
        logger.info(x_model_pipeline.ops.summarize())

        # get model transforms
        train_transform = x_model_pipeline.config.model_pipeline_config.train_transform
        eval_transform = x_model_pipeline.config.model_pipeline_config.eval_transform
        if dataset.train is not None:
            dataset.train.output_transform = train_transform
        if dataset.validation is not None:
            dataset.validation.output_transform = eval_transform
        if dataset.test is not None:
            dataset.test.output_transform = eval_transform

        # build data pipeline
        data_pipeline = DataPipeline(dataset=dataset)

        return ModelExplainerState(
            data_pipeline=data_pipeline,
            x_model_pipeline=x_model_pipeline,
            tb_logger=tb_logger,
        )

    def run(self) -> None:
        explanation_engine = self._build_explanation_engine()
        state = explanation_engine.run()
        print("state", state.output)
        metrics = _format_metrics_for_logging(state.metrics)
        logger.info("Final explanation metrics:")
        logger.info(json.dumps(metrics, indent=4))

        # serialize test metrics
        self._config.dump_metrics_file(data=metrics)  #  type: ignore
        return metrics
