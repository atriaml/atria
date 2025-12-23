from __future__ import annotations

from collections import OrderedDict
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
from torchxai.data_types import BatchExplanationTarget

from atria_insights.model_pipelines._common import (
    ExplainableModelPipelineConfig,
    ExplanationTargetStrategy,
)
from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline
from atria_insights.model_pipelines._registry_groups import EXPLAINABLE_MODEL_PIPELINES

logger = get_logger(__name__)


def _get_first_layer(module, name=None):
    children = list(module.named_children())
    if len(children) > 0:
        return _get_first_layer(
            children[0][1],
            name=children[0][0] if name is None else name + "." + children[0][0],
        )
    return name, module


class ExplainableImageModelPipelineConfig(ExplainableModelPipelineConfig):
    model_pipeline: ImageModelPipelineConfig


T_ExplainableImageModelPipelineConfig = TypeVar(
    "T_ExplainableImageModelPipelineConfig", bound="ExplainableImageModelPipelineConfig"
)


class ExplainableImageModelPipeline(
    ExplainableModelPipeline[
        ExplainableImageModelPipelineConfig,
        ImageTensorDataModel | DocumentTensorDataModel,
    ]
):
    __abstract__ = True
    __config__ = ExplainableImageModelPipelineConfig

    def __init__(
        self, config: ExplainableImageModelPipelineConfig, labels: DatasetLabels
    ) -> None:
        super().__init__(config=config, labels=labels)
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

    def _explained_inputs(
        self, batch: ImageTensorDataModel | DocumentTensorDataModel
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        assert batch.image is not None, "Input images are required for explanation."
        return batch.image


class ExplainableImageClassificationPipelineConfig(ExplainableImageModelPipelineConfig):
    model_pipeline: ImageClassificationPipelineConfig = (
        ImageClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "image_classification"


@EXPLAINABLE_MODEL_PIPELINES.register("image_classification")
class ExplainableImageClassificationPipeline(ExplainableImageModelPipeline):
    __config__ = ExplainableImageClassificationPipelineConfig
