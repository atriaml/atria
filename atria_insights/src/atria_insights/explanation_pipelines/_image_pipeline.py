from __future__ import annotations

from typing import TypeVar

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._image_pipeline import (
    ImageClassificationPipelineConfig,
    ImageModelPipeline,
    ImageModelPipelineConfig,
)
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_transforms.data_types._image import ImageTensorDataModel
from atria_types._datasets import DatasetLabels

from atria_insights.data_types._targets import BatchExplanationTarget
from atria_insights.explanation_pipelines._base import BaseExplanationPipeline
from atria_insights.explanation_pipelines._common import (
    ExplanationPipelineConfig,
    ExplanationTargetStrategy,
)
from atria_insights.explanation_pipelines._registry_groups import EXPLANATION_PIPELINES

logger = get_logger(__name__)


class ImageModelExplanationPipelineConfig(ExplanationPipelineConfig):
    pass 

T_ImageModelExplanationPipelineConfig = TypeVar(
    "T_ImageModelExplanationPipelineConfig", bound="ImageModelExplanationPipelineConfig"
)


class ImageModelExplanationPipeline(
    BaseExplanationPipeline[
        ImageModelExplanationPipelineConfig,
        ImageTensorDataModel | DocumentTensorDataModel,
    ]
):
    __abstract__ = True
    __config__ = ImageModelExplanationPipelineConfig

    def __init__(
        self,
        config: ImageModelExplanationPipelineConfig,
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

    def _target(
        self,
        batch: ImageTensorDataModel | DocumentTensorDataModel,
        model_outputs: torch.Tensor,
    ) -> BatchExplanationTarget | list[BatchExplanationTarget]:
        assert self._model_pipeline._labels.classification is not None, (
            "Labels are required for explanation target strategies other than 'predicted'."
        )
        label_names = self._model_pipeline._labels.classification
        if (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.ground_truth
        ):
            assert batch.label is not None, (
                "Ground truth labels are required for explanation target strategies other than 'predicted'."
            )
            return BatchExplanationTarget(
                value=batch.label.tolist(),
                name=[label_names[idx] for idx in batch.label.tolist()],
            )
        elif (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.predicted
        ):
            predictions = model_outputs.argmax(dim=-1)
            prediction_label_names = [label_names[idx] for idx in predictions]
            return BatchExplanationTarget(
                value=predictions.tolist(), name=prediction_label_names
            )
        else:
            # in case of 'all' we compute the explanations for all classes
            total_labels = model_outputs.shape[1]
            batch_size = model_outputs.shape[0]
            return [
                BatchExplanationTarget(
                    value=[label_index for _ in range(batch_size)],
                    name=[label_names[label_index] for _ in range(batch_size)],
                )
                for label_index in range(total_labels)
            ]

    def _explained_inputs(  # type: ignore[override]
        self, batch: ImageTensorDataModel | DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        assert batch.image is not None, "Input images are required for explanation."
        return {"image": batch.image}


class ImageClassificationExplanationPipelineConfig(ImageModelExplanationPipelineConfig):
    @property
    def name(self) -> str:
        return "image_classification"


@EXPLANATION_PIPELINES.register("image_classification")
class ImageClassificationExplanationPipeline(ImageModelExplanationPipeline):
    __config__ = ImageClassificationExplanationPipelineConfig
