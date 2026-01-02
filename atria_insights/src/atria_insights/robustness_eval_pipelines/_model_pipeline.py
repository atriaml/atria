from __future__ import annotations

import inspect
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generic

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._ops import ModelPipelineOps
from atria_models.core.types.model_outputs import ModelOutput
from atria_registry._module_base import ConfigurableModule
from atria_transforms.core._data_types._base import T_TensorDataModel
from atria_types._datasets import DatasetLabels

from atria_insights.baseline_generators import SequenceBaselineGeneratorConfig
from atria_insights.robustness_eval_pipelines._common import (
    T_RobustnessEvalModelPipelineConfig,
)

logger = get_logger(__name__)


class RobustnessEvalModelPipeline(
    ConfigurableModule[T_RobustnessEvalModelPipelineConfig],
    Generic[T_RobustnessEvalModelPipelineConfig, T_TensorDataModel],
):
    __abstract__ = True
    __config__: type[T_RobustnessEvalModelPipelineConfig]

    def __init__(
        self,
        config: T_RobustnessEvalModelPipelineConfig,
        labels: DatasetLabels,
        persist_to_disk: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self._persist_to_disk = persist_to_disk
        self._cache_dir = cache_dir
        if self._persist_to_disk and not self._cache_dir:
            raise ValueError("cache_dir must be specified if persist_to_disk is True.")

        self._model_pipeline = self.config.model_pipeline.build(labels=labels)

        # build model with wrapped forward
        self._model_signature = inspect.signature(self._model_pipeline._model.forward)
        self._wrapped_model = self._wrap_model_forward(self._model_pipeline._model)

        # build baselines generator
        if isinstance(self.config.baseline_generator, SequenceBaselineGeneratorConfig):
            raise ValueError(
                "SequenceBaselineGeneratorConfig is not supported here. "
                "Please use FeatureBasedBaselineGeneratorConfig or SimpleBaselineGeneratorConfig."
            )
        self._baseline_generator = self.config.baseline_generator.build(
            model=self._model_pipeline._model
        )

    @property
    def ops(self) -> Any:
        return ModelPipelineOps(self._model_pipeline)

    def _dump_config(self, config_dir: Path) -> dict:
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "config.yaml", "w") as f:
            f.write(self._config.to_yaml())
            return self._config.model_dump()

    def summarize(self):
        logger.info("Robustness Evaluation Model Pipeline Summary:")
        logger.info(self._model_pipeline.ops.summarize())
        logger.info("Baseline Generator Config: %s", self.config.baseline_generator)

    @abstractmethod
    def _features(self, batch: T_TensorDataModel) -> OrderedDict[str, torch.Tensor]:
        """Prepare the input features for the explainer."""
        pass

    def _additional_forward_kwargs(
        self, batch: T_TensorDataModel
    ) -> OrderedDict[str, Any] | None:
        """Prepare additional forward kwargs for the model."""
        return None

    def _baselines(
        self, features: OrderedDict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        """Generate baselines for the explainer."""
        features_tuple = tuple(f.value for f in features.values())
        baselines = self._baseline_generator(features_tuple)
        return OrderedDict(
            (key, baseline)
            for key, baseline in zip(features.keys(), baselines, strict=True)
        )

    @abstractmethod
    def robustness_eval_step(self, batch: T_TensorDataModel) -> ModelOutput:
        pass
        # # prepare explained inputs
        # inputs = self._features(batch)

        # # prepare additional forward args
        # additional_forward_kwargs = (
        #     self._additional_forward_kwargs(batch) or OrderedDict()
        # )

        # # prepare baselines
        # baselines = self._baselines(explained_inputs=inputs)

        # # forward pass
        # model_outputs = self._wrapped_model(batch)
        # return model_outputs
