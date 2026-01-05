from __future__ import annotations

from abc import abstractmethod
from collections import dict
from pathlib import Path
from typing import Any, Generic

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._ops import ModelPipelineOps
from atria_models.core.model_pipelines.utilities import log_tensor_info
from atria_models.core.types.model_outputs import ModelOutput
from atria_registry._module_base import ConfigurableModule
from atria_transforms.core._data_types._base import T_TensorDataModel
from atria_types._datasets import DatasetLabels
from ignite.metrics import Metric

from atria_insights.feature_perturbation.pipelines._config import (
    T_FeaturePerturbationPipelineConfig,
)

logger = get_logger(__name__)


class FeaturePerturbationPipeline(
    ConfigurableModule[T_FeaturePerturbationPipelineConfig],
    Generic[T_FeaturePerturbationPipelineConfig, T_TensorDataModel],
):
    __abstract__ = True
    __config__: type[T_FeaturePerturbationPipelineConfig]

    def __init__(
        self,
        config: T_FeaturePerturbationPipelineConfig,
        labels: DatasetLabels,
        persist_to_disk: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self._labels = labels
        self._persist_to_disk = persist_to_disk
        self._cache_dir = cache_dir
        self._build()
        if self._persist_to_disk and not self._cache_dir:
            raise ValueError("cache_dir must be specified if persist_to_disk is True.")

    @property
    def ops(self) -> Any:
        return ModelPipelineOps(self._model_pipeline)

    def summarize(self):
        logger.info("Perturbation Robustness Evaluation Pipeline Summary:")
        logger.info(self._model_pipeline.ops.summarize())
        logger.info("Feature Segmentor Config: %s", self.config.feature_segmentor)
        logger.info("Baseline Generator Config: %s", self.config.baseline_generator)

    def _dump_config(self, config_dir: Path) -> dict:
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "config.yaml", "w") as f:
            f.write(self._config.to_yaml())
            return self._config.model_dump()

    def _build_model_pipeline(self):
        return self.config.model_pipeline.build(labels=self._labels)

    def _build_feature_segmentor(self):
        return self.config.feature_segmentor.build()

    def _build_baseline_generator(self):
        return self.config.baseline_generator.build(model=self._model_pipeline._model)

    def _build(self):
        self._model_pipeline = self._build_model_pipeline()
        self._feature_segmentor = self._build_feature_segmentor()
        self._baseline_generator = self._build_baseline_generator()

    def _perturbed_inputs(
        self, batch: T_TensorDataModel, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Prepare the input features for the explainer."""
        pass

    def _additional_forward_kwargs(
        self, batch: T_TensorDataModel
    ) -> dict[str, Any] | None:
        """Prepare additional forward kwargs for the model."""
        return None

    def _baselines(
        self, explained_inputs: torch.Tensor | dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Generate baselines for the explainer."""
        logger.debug(
            "Generating baselines using baseline generator with config: %s",
            self.config.baseline_generator,
        )
        baselines = self._baseline_generator(explained_inputs, **kwargs)
        log_tensor_info(baselines, name="baselines")
        return baselines

    def _feature_mask(
        self, explained_inputs: torch.Tensor | dict[str, torch.Tensor], **kwargs
    ) -> Any:
        """Generate feature mask using the feature segmentor."""
        logger.debug(
            "Generating feature mask using feature segmentor with config: %s",
            self.config.feature_segmentor,
        )
        feature_masks = self._feature_segmentor(explained_inputs, **kwargs)
        log_tensor_info(feature_masks, name="feature_masks")
        return feature_masks

    @abstractmethod
    def evaluation_step(  # type: ignore[override]
        self, perturbed_batch: dict[str, Any], additional_forward_kwargs: dict[str, Any]
    ) -> ModelOutput:
        raise NotImplementedError()

    def _perturb_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        baselines: dict[str, torch.Tensor],
        feature_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(iter(inputs.values())).device

        # if no feature masks provided, treat each element as its own feature
        if feature_masks is None:
            # each element is its own feature
            feature_masks = {
                k: torch.arange(v.numel(), device=device).reshape(v.shape)
                for k, v in inputs.items()
            }

        # collect feature indices
        feature_values = torch.cat([fm.flatten() for fm in feature_masks.values()])
        min_feature_idx = int(feature_values.min())
        max_feature_idx = int(feature_values.max())

        feature_indices = torch.arange(
            min_feature_idx, max_feature_idx + 1, device=device
        )

        # sample features to perturb
        total_features_perturbed = max(
            1, int(self.config.percent_features_perturbed * len(feature_indices))
        )
        rand_indices = torch.randperm(len(feature_indices), device=device)[
            :total_features_perturbed
        ]
        selected_feature_indices = feature_indices[rand_indices]

        # apply perturbation
        perturbed_inputs = {}
        for key, fm in feature_masks.items():
            mask = torch.isin(fm, selected_feature_indices)
            perturbed_inputs[key] = inputs[key] * (~mask) + baselines[key] * mask

        return perturbed_inputs

    def perturb_and_evaluate(self, batch: T_TensorDataModel) -> ModelOutput:
        """Prepare the inputs for the explainer step."""
        with torch.no_grad():
            # prepare explained inputs
            inputs = self._perturbed_inputs(batch)

            # prepare additional forward args
            additional_forward_kwargs = self._additional_forward_kwargs(batch) or dict()

            # prepare baselines
            baselines = self._baselines(explained_inputs=inputs)

            # prepare feature mask
            feature_mask = self._feature_mask(explained_inputs=inputs)

            # perturb the inputs
            inputs = self._perturb_inputs(
                inputs=inputs, baselines=baselines, feature_masks=feature_mask
            )

            # forward pass
            return self.evaluation_step(
                perturbed_batch=inputs,
                additional_forward_kwargs=additional_forward_kwargs,
            )

    def build_metrics(self, device: torch.device | str = "cpu") -> dict[str, Metric]:
        return self._model_pipeline.build_metrics(stage="test", device=device)
