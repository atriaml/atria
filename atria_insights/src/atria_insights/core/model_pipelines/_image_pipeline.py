from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._image_pipeline import ImageModelPipeline
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_transforms.data_types._image import ImageTensorDataModel
from torchxai.data_types import (
    ExplanationTargetType,
    SingleTargetAcrossBatch,
    SingleTargetPerSample,
)

from atria_insights.core.model_pipelines._common import (
    ExplainableModelPipelineConfig,
    ExplanationTargetStrategy,
)
from atria_insights.core.model_pipelines._model_pipeline import ExplainableModelPipeline
from atria_insights.core.model_pipelines._registry_groups import (
    EXPLAINABLE_MODEL_PIPELINE,
)

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
    pass


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

    def _model_forward(
        self, batch: ImageTensorDataModel | DocumentTensorDataModel
    ) -> torch.Tensor:
        from torch.nn.functional import softmax

        model_outputs = super()._model_forward(batch)
        if isinstance(model_outputs, dict):
            logits = model_outputs["logits"]
        elif hasattr(model_outputs, "logits"):
            logits = model_outputs.logits
        else:
            logits = model_outputs
        return softmax(logits, dim=-1)

    def _target(
        self,
        batch: ImageTensorDataModel | DocumentTensorDataModel,
        model_outputs: torch.Tensor,
    ) -> ExplanationTargetType | list[ExplanationTargetType]:
        if (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.ground_truth
        ):
            assert batch.label is not None, (
                "Ground truth labels are required for explanation target strategies other than 'predicted'."
            )
            return SingleTargetPerSample(indices=batch.label.tolist())
        elif self.config.explanation_target_strategy == ExplanationTargetStrategy.all:
            predictions = model_outputs.argmax(dim=-1)
            return SingleTargetPerSample(indices=predictions.tolist())
        else:
            # in case of 'all' we compute the explanations for all classes
            total_labels = model_outputs.shape[1]
            return [
                SingleTargetAcrossBatch(index=label_index)
                for label_index in range(total_labels)
            ]

    def _explained_inputs(
        self, batch: ImageTensorDataModel | DocumentTensorDataModel
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        return batch.image


@EXPLAINABLE_MODEL_PIPELINE.register("image_classification")
class ExplainableImageClassificationPipeline(ImageModelPipeline):
    __config__ = ExplainableImageModelPipelineConfig
