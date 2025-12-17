from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._sequence_pipeline import SequenceModelPipeline
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_types._datasets import DatasetLabels
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


class ExplainableSequenceModelPipelineConfig(ExplainableModelPipelineConfig):
    pass


T_ExplainableSequenceModelPipelineConfig = TypeVar(
    "T_ExplainableSequenceModelPipelineConfig",
    bound="ExplainableSequenceModelPipelineConfig",
)


class ExplainableSequenceModelPipeline(
    ExplainableModelPipeline[
        ExplainableSequenceModelPipelineConfig, DocumentTensorDataModel
    ]
):
    __abstract__ = True
    __config__ = ExplainableSequenceModelPipelineConfig

    def __init__(
        self, config: ExplainableSequenceModelPipelineConfig, labels: DatasetLabels
    ) -> None:
        super().__init__(config=config, labels=labels)
        assert isinstance(self._model_pipeline, SequenceModelPipeline), (
            f"{self.__class__.__name__} can only be used with SequenceModelPipeline. Found {self._model_pipeline=}"
        )

    def _model_forward(self, batch: DocumentTensorDataModel) -> torch.Tensor:
        from torch.nn.functional import softmax

        model_outputs = self._model_pipeline._model(batch.Sequence)
        if isinstance(model_outputs, dict):
            logits = model_outputs["logits"]
        elif hasattr(model_outputs, "logits"):
            logits = model_outputs.logits
        else:
            logits = model_outputs
        return softmax(logits, dim=-1)

    def _target(
        self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
    ) -> ExplanationTargetType | list[ExplanationTargetType]:
        if (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.ground_truth
        ):
            assert batch.label is not None, (
                "Ground truth labels are required for explanation target strategies other than 'predicted'."
            )
            return SingleTargetPerSample(indices=batch.label.tolist())
        elif (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.predicted
        ):
            predictions = model_outputs.argmax(dim=-1)
            return SingleTargetPerSample(indices=predictions.tolist())
        else:
            # in case of 'all' we compute the explanations for all classes
            total_labels = model_outputs.shape[1]
            return [
                SingleTargetAcrossBatch(index=label_index)
                for label_index in range(total_labels)
            ]


@EXPLAINABLE_MODEL_PIPELINE.register("sequence_classification")
class ExplainableSequenceClassificationPipeline(ExplainableSequenceModelPipeline):
    __config__ = ExplainableSequenceModelPipelineConfig

    def _explained_inputs(
        self, batch: DocumentTensorDataModel
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        inputs = self.prepare_explainable_inputs(*args, **kwargs)
        baselines = self.prepare_baselines_from_inputs(*args, **kwargs)
        metric_baselines = self.prepare_metric_baselines_from_inputs(*args, **kwargs)
        feature_masks, total_features, frozen_features = (
            self.prepare_feature_masks_from_inputs(*args, **kwargs)
        )
        feature_masks = self.expand_feature_masks_to_explainable_inputs(
            inputs, feature_masks
        )
        additional_forward_kwargs = self.prepare_additional_forward_kwargs(
            *args, **kwargs
        )
        constant_shifts = self.prepare_constant_shifts(*args, **kwargs)
        input_layer_names = self.prepare_input_layer_names()

        return ExplainerArguments(
            inputs=inputs,
            baselines=baselines,
            metric_baselines=metric_baselines,
            feature_masks=feature_masks,
            total_features=total_features,
            additional_forward_kwargs=(
                {} if additional_forward_kwargs is None else additional_forward_kwargs
            ),
            constant_shifts=constant_shifts,
            input_layer_names=input_layer_names,
            frozen_features=frozen_features,
        )
