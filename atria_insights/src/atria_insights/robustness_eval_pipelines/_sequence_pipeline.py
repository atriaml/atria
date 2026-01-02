from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Any, TypeVar

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._sequence_pipeline import (
    QuestionAnsweringPipelineConfig,
    SequenceClassificationPipelineConfig,
    SequenceModelPipeline,
    TokenClassificationPipelineConfig,
)
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_types._datasets import DatasetLabels
from pydantic import model_validator
from torchxai.data_types import (
    ExplanationTargetType,
    SingleTargetAcrossBatch,
    SingleTargetPerSample,
)

from atria_insights.feature_segmentors._sequence import (
    FeatureSegmentorConfigType,
    SequenceFeatureMaskSegmentorConfig,
)
from atria_insights.model_pipelines._common import (
    ExplainableModelPipelineConfig,
    ExplanationTargetStrategy,
)
from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline
from atria_insights.model_pipelines._registry_groups import EXPLAINABLE_MODEL_PIPELINES
from atria_insights.model_pipelines.baseline_generators import (
    BaselineGeneratorConfigType,
)
from atria_insights.model_pipelines.baseline_generators._sequence import (
    SequenceBaselineGeneratorConfig,
)

logger = get_logger(__name__)


class ExplainableSequenceModelPipelineConfig(ExplainableModelPipelineConfig):
    feature_segmentor: FeatureSegmentorConfigType = SequenceFeatureMaskSegmentorConfig()
    baseline_generator: BaselineGeneratorConfigType = SequenceBaselineGeneratorConfig()

    @model_validator(mode="after")
    def validate_configs(self) -> ExplainableSequenceModelPipelineConfig:
        if not isinstance(self.feature_segmentor, SequenceFeatureMaskSegmentorConfig):
            raise ValueError(
                "feature_segmentor must be an instance of SequenceFeatureMaskSegmentorConfig"
            )
        if not isinstance(self.baseline_generator, SequenceBaselineGeneratorConfig):
            raise ValueError(
                "baseline_generator must be an instance of SequenceBaselineGeneratorConfig"
            )
        return self


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

        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)
        self._input_ids_list = []
        for param in inspect.signature(
            self._model_pipeline._model.ids_to_embeddings
        ).parameters.values():
            if param.name != "self":
                self._input_ids_list.append(param.name)

    def _explained_inputs(
        self, batch: DocumentTensorDataModel
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)

        inputs = {
            "token_ids": batch.token_ids,
            "token_type_ids": batch.token_type_ids,
            "position_ids": None,
        }

        if "image" in self._input_ids_list and self._model_pipeline.config.use_image:
            assert batch.image is not None, "Image cannot be None"
            inputs["image"] = batch.image

        if (
            "token_bboxes" in self._input_ids_list
            and self._model_pipeline.config.use_bbox
        ):
            assert batch.token_bboxes is not None, "Token bboxes cannot be None"
            token_bboxes = batch.token_bboxes
            if batch.metadata.bbox_normalized[0] and token_bboxes is not None:
                token_bboxes = (
                    (token_bboxes * 1000.0).clip(0, 1000).long()
                    if token_bboxes is not None
                    else None
                )
            inputs["token_bboxes"] = token_bboxes

        # generate input embeddings
        return self._model_pipeline._model.ids_to_embeddings(**inputs).to_ordered_dict()

    def _additional_forward_args(self, batch: DocumentTensorDataModel):
        return {"attention_mask": batch.attention_mask, "is_embedding": True}

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


class ExplainableSequenceClassificationPipelineConfig(
    ExplainableSequenceModelPipelineConfig
):
    model_pipeline: SequenceClassificationPipelineConfig = (
        SequenceClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "sequence_classification"


@EXPLAINABLE_MODEL_PIPELINES.register("sequence_classification")
class ExplainableSequenceClassificationPipeline(ExplainableSequenceModelPipeline):
    __config__ = ExplainableSequenceClassificationPipelineConfig


class ExplainableTokenClassificationPipelineConfig(
    ExplainableSequenceModelPipelineConfig
):
    model_pipeline: TokenClassificationPipelineConfig = (
        TokenClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "token_classification"


@EXPLAINABLE_MODEL_PIPELINES.register("token_classification")
class ExplainableTokenClassificationPipeline(ExplainableSequenceModelPipeline):
    __config__ = ExplainableTokenClassificationPipelineConfig

    def _target(
        self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
    ) -> ExplanationTargetType | list[ExplanationTargetType]:
        if (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.ground_truth
        ):
            raise ValueError(
                "Ground truth explanation targets are not supported for token classification tasks."
            )
        elif (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.predicted
        ):
            # model_outputs is returned predictions of shape [batch_size, seq_len] tensor so a target for each token
            return [
                SingleTargetAcrossBatch(index=i) for i in range(model_outputs.shape[1])
            ]
        else:
            raise ValueError(
                "Only 'predicted' explanation target strategy is supported for token classification tasks."
            )

    def _model_forward(
        self,
        explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
        additional_forward_args: dict[str, Any] | None,
    ) -> torch.Tensor:
        from torch.nn.functional import softmax

        model_outputs = self._model_pipeline._model(
            explained_inputs, **(additional_forward_args or {})
        )
        if isinstance(model_outputs, dict):
            logits = model_outputs["logits"]
        elif hasattr(model_outputs, "logits"):
            logits = model_outputs.logits
        else:
            logits = model_outputs
        return softmax(logits, dim=-1)

    def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
        class WrappedModel(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self._model = model

            def forward(
                self,
                explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
                additional_forward_args: dict[str, Any] | None = None,
            ) -> torch.Tensor:
                model_outputs = self._model(
                    explained_inputs, **(additional_forward_args or {})
                )
                logits = (
                    model_outputs.logits
                    if hasattr(model_outputs, "logits")
                    else model_outputs
                )
                probs = self.softmax(logits)
                if len(probs.shape) == 3:
                    probs = torch.gather(
                        probs, 2, probs.argmax(dim=-1).unsqueeze(-1)
                    ).squeeze(-1)
                return probs

        return WrappedModel(model)


class ExplainableQuestionAnsweringPipelineConfig(
    ExplainableSequenceModelPipelineConfig
):
    model_pipeline: QuestionAnsweringPipelineConfig = QuestionAnsweringPipelineConfig()

    @property
    def name(self) -> str:
        return "question_answering"


@EXPLAINABLE_MODEL_PIPELINES.register("question_answering")
class ExplainableQuestionAnsweringPipeline(ExplainableSequenceModelPipeline):
    __config__ = ExplainableQuestionAnsweringPipelineConfig

    def _target(
        self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
    ) -> ExplanationTargetType | list[ExplanationTargetType]:
        if (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.ground_truth
        ):
            raise ValueError(
                "Ground truth explanation targets are not supported for token classification tasks."
            )
        elif (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.predicted
        ):
            # model_outputs is returned predictions of shape [batch_size, seq_len] tensor so a target for each token
            return [
                SingleTargetAcrossBatch(index=i) for i in range(1)
            ]  # for q/a we explain the start token only
        else:
            raise ValueError(
                f"Explanation target strategy {self.config.explanation_target_strategy} is not supported for question answering tasks."
            )

    def _model_forward(
        self,
        explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
        additional_forward_args: dict[str, Any] | None,
    ) -> torch.Tensor:
        from torch.nn.functional import softmax

        model_outputs = self._model_pipeline._model(
            explained_inputs, **(additional_forward_args or {})
        )
        if isinstance(model_outputs, dict):
            logits = model_outputs["logits"]
        elif hasattr(model_outputs, "logits"):
            logits = model_outputs.logits
        else:
            logits = model_outputs
        return softmax(logits, dim=-1)

    def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
        class WrappedModel(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self._model = model

            def forward(
                self,
                explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
                additional_forward_args: dict[str, Any] | None = None,
            ) -> torch.Tensor:
                from torch.nn.functional import softmax

                model_outputs = self._model(
                    explained_inputs, **(additional_forward_args or {})
                )
                start_pred = softmax(model_outputs.start_logits, dim=-1)
                end_pred = softmax(model_outputs.end_logits, dim=-1)
                return torch.cat([start_pred, end_pred])

        return WrappedModel(model)
