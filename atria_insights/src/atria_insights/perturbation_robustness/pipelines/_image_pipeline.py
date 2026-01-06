from __future__ import annotations

from typing import Any, TypeVar

import torch
from atria_insights.feature_perturbation._registry_groups import (
    FEATURE_PERTURBATION_PIPELINES,
)
from atria_insights.feature_perturbation.pipelines._base import (
    PerturbationRobustnessPipeline,
)
from atria_insights.feature_perturbation.pipelines._config import (
    PerturbationRobustnessPipelineConfig,
)
from atria_logger import get_logger
from atria_models.core.model_pipelines._image_pipeline import (
    ImageClassificationPipeline,
    ImageClassificationPipelineConfig,
    ImageModelPipeline,
    ImageModelPipelineConfig,
)
from atria_models.core.types.model_outputs import ModelOutput
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_transforms.data_types._image import ImageTensorDataModel
from atria_types._datasets import DatasetLabels

from atria_insights.baseline_generators._simple import SimpleBaselineGeneratorConfig

logger = get_logger(__name__)


class ImagePerturbationRobustnessPipelineConfig(PerturbationRobustnessPipelineConfig):
    model_pipeline: ImageModelPipelineConfig
    baseline_generator: SimpleBaselineGeneratorConfig = SimpleBaselineGeneratorConfig()


T_ImagePerturbationRobustnessPipelineConfig = TypeVar(
    "T_ImagePerturbationRobustnessPipelineConfig",
    bound="ImagePerturbationRobustnessPipelineConfig",
)


class ImagePerturbationRobustnessPipeline(
    PerturbationRobustnessPipeline[
        T_ImagePerturbationRobustnessPipelineConfig,
        ImageTensorDataModel | DocumentTensorDataModel,
    ]
):
    __abstract__ = True

    def __init__(
        self,
        config: T_ImagePerturbationRobustnessPipelineConfig,
        labels: DatasetLabels,
        persist_to_disk: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            labels=labels,
            persist_to_disk=persist_to_disk,
            cache_dir=cache_dir,
        )
        assert isinstance(self._model_pipeline, ImageModelPipeline), (
            f"{self.__class__.__name__} can only be used with ImageModelPipeline. Found {self._model_pipeline=}"
        )

    def _perturbed_inputs(
        self, batch: ImageTensorDataModel | DocumentTensorDataModel, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Prepare the input features for the explainer."""
        return {"images": batch.image}

    def _perturb_inputs(
        inputs: dict[str, torch.Tensor],
        baselines: dict[str, torch.Tensor],
        feature_masks: dict[str, torch.Tensor] | None = None,
        masking_probability: float = 1.0,
    ) -> None:
        """Perturb the inputs in-place based on baselines and feature masks."""
        perturbed_inputs = {}
        for key in inputs:
            input_tensor = inputs[key]
            baseline_tensor = baselines[key]
            if feature_masks and key in feature_masks:
                feature_mask = feature_masks[key]
                mask = torch.bernoulli(masking_probability * feature_mask.float()).to(
                    input_tensor.device
                )
                perturbed_input = input_tensor * (1 - mask) + baseline_tensor * mask
            else:
                mask = torch.bernoulli(
                    torch.full_like(input_tensor, masking_probability)
                ).to(input_tensor.device)
                perturbed_input = input_tensor * (1 - mask) + baseline_tensor * mask
            perturbed_inputs[key] = perturbed_input
        return perturbed_inputs


class ImageClassificationPerturbationRobustnessPipelineConfig(
    ImagePerturbationRobustnessPipelineConfig
):
    model_pipeline: ImageClassificationPipelineConfig = (
        ImageClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "image_classification"


@FEATURE_PERTURBATION_PIPELINES.register("image_classification")
class ImageClassificationPerturbationRobustnessPipeline(
    ImagePerturbationRobustnessPipeline
):
    __config__ = ImageClassificationPerturbationRobustnessPipelineConfig

    def evaluation_step(
        self,
        batch: ImageTensorDataModel | DocumentTensorDataModel,
        perturbed_inputs: dict[str, Any],
        additional_forward_kwargs: dict[str, Any],
    ) -> ModelOutput:
        assert isinstance(self._model_pipeline, ImageClassificationPipeline), (
            f"{self.__class__.__name__} can only be used with ImageClassificationPipeline. Found {self._model_pipeline=}"
        )
