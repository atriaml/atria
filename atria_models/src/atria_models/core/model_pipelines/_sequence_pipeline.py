from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from atria_logger import get_logger
from atria_metrics.core.question_answering.due_eval import DueEvalMetricConfig
from atria_metrics.registry.classification import (
    AccuracyMetricConfig,
    ConfusionMatrixMetricConfig,
    F1ScoreMetricConfig,
    PrecisionMetricConfig,
    RecallMetricConfig,
)
from atria_metrics.registry.entity_labeling import (
    LayoutF1MacroMetricConfig,
    LayoutF1MetricConfig,
    LayoutPrecisionMacroMetricConfig,
    LayoutPrecisionMetricConfig,
    LayoutRecallMacroMetricConfig,
    LayoutRecallMetricConfig,
    SeqEvalMetricConfig,
)
from atria_metrics.registry.question_answering.due_eval import DocVQAEvalConfig
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_transforms.tfs._document_processor._task_tfs import (
    QuestionAnsweringDocumentProcessor,
    SequenceClassificationDocumentProcessor,
    TokenClassificationDocumentProcessor,
)
from atria_transforms.tfs._hf_processor import HuggingfaceProcessor
from pydantic import Field, model_validator
from torch._tensor import Tensor

from atria_models.core.model_builders._common import ModelBuilderType
from atria_models.core.model_pipelines._common import ModelConfig, ModelPipelineConfig
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from atria_models.core.model_pipelines.utilities import (
    _postprocess_qa_predictions,
    log_tensors_debug_info,
)
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_models.core.models.transformers._outputs import (
    SequenceClassificationHeadOutput,
    TransformersEncoderModelOutput,
)
from atria_models.core.types.model_outputs import (
    ClassificationModelOutput,
    LayoutTokenClassificationModelOutput,
    ModelOutput,
    QAModelOutput,
    QAPair,
    TokenClassificationModelOutput,
)
from atria_models.registry.registry_groups import MODEL_PIPELINES

if TYPE_CHECKING:
    import torch
    from ignite.metrics import Metric

logger = get_logger(__name__)


class SequenceModelPipelineConfig(ModelPipelineConfig):
    use_bbox: bool = True
    use_image: bool = True
    use_segment_info: bool = True
    input_stride: int = 0


class SequenceModelPipeline(ModelPipeline[SequenceModelPipelineConfig]):
    __config__ = SequenceModelPipelineConfig

    def training_step(  # type: ignore[override]
        self, batch: DocumentTensorDataModel, **kwargs
    ) -> ModelOutput:
        inputs = self._input_transform(batch, require_labels=True)
        log_tensors_debug_info(inputs, title="train_inputs")
        model_output = self._model(**inputs)
        log_tensors_debug_info(model_output, title="train_model_output")
        loss = self._compute_loss(model_output=model_output, batch=batch)
        return self._output_transform(loss=loss, model_output=model_output, batch=batch)

    def evaluation_step(  # type: ignore[override]
        self,
        batch: DocumentTensorDataModel,
        stage: Literal["validation", "test"],
        **kwargs,
    ) -> ModelOutput:
        inputs = self._input_transform(batch, require_labels=True)
        log_tensors_debug_info(inputs, title=f"{stage}_inputs")
        model_output = self._model(**inputs)
        log_tensors_debug_info(model_output, title=f"{stage}_model_output")
        loss = self._compute_loss(model_output=model_output, batch=batch)
        return self._output_transform(loss=loss, model_output=model_output, batch=batch)

    def predict_step(  # type: ignore[override]
        self, batch: DocumentTensorDataModel, **kwargs
    ) -> ModelOutput:
        inputs = self._input_transform(batch, require_labels=False)
        log_tensors_debug_info(inputs, title="predict_inputs")
        model_output = self._model(**inputs)
        log_tensors_debug_info(model_output, title="predict_model_output")
        return self._output_transform(loss=None, model_output=model_output, batch=batch)

    def _model_build_kwargs(self) -> dict[str, object]:
        raise NotImplementedError(
            "_model_build_kwargs must be implemented in subclasses."
        )

    def _compute_loss(
        self, model_output: Any, batch: DocumentTensorDataModel
    ) -> torch.Tensor:
        raise NotImplementedError("_compute_loss must be implemented in subclasses.")

    def _input_transform(
        self, batch: DocumentTensorDataModel, require_labels: bool = False
    ) -> dict[str, torch.Tensor | None]:
        if isinstance(self._model, TransformersEncoderModel):
            inputs = {
                "token_id_or_embeddings": batch.token_ids,
                "token_type_ids_or_embeddings": batch.token_type_ids,
                "attention_mask": batch.attention_mask,
            }
        else:
            inputs = {
                "input_ids": batch.token_ids,
                "token_type_ids": batch.token_type_ids,
                "attention_mask": batch.attention_mask,
            }

        if "pixel_values" in self._model_args_list and self.config.use_image:
            assert batch.image is not None, "Image cannot be None"
            if isinstance(self._model, TransformersEncoderModel):
                inputs["image"] = batch.image
            else:
                inputs["pixel_values"] = batch.image

        if "bbox" in self._model_args_list and self.config.use_bbox:
            assert batch.token_bboxes is not None, "Token bboxes cannot be None"
            token_bboxes = batch.token_bboxes
            if batch.metadata.bbox_normalized[0] and token_bboxes is not None:
                token_bboxes = (
                    (token_bboxes * 1000.0).clip(0, 1000).long()
                    if token_bboxes is not None
                    else None
                )
            if isinstance(self._model, TransformersEncoderModel):
                inputs["layout_ids"] = token_bboxes
            else:
                inputs["bbox"] = token_bboxes

        if "segment_index" in self._model_args_list and self.config.use_segment_info:
            inputs.update(
                {
                    "segment_index": batch.segment_index,
                    "segment_inner_token_rank": batch.segment_inner_token_rank,
                    "first_token_idxes": batch.first_token_idxes,
                    "first_token_idxes_mask": batch.first_token_idxes_mask,
                }
            )

        if require_labels:
            inputs.update(self._prepare_labels(batch=batch))

        return inputs

    def _prepare_labels(
        self, batch: DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        return {}

    def _output_transform(
        self,
        loss: torch.Tensor | None,
        model_output: Any,
        batch: DocumentTensorDataModel,
    ) -> ModelOutput:
        raise NotImplementedError(
            "_output_transform must be implemented in subclasses."
        )


class SequenceClassificationPipelineConfig(SequenceModelPipelineConfig):
    model: ModelConfig = ModelConfig(
        model_name_or_path="bert-base-uncased",
        builder_type=ModelBuilderType.transformers,
        model_type="sequence_classification",
    )
    name: str = "sequence_classification"
    metrics: (
        list[
            Annotated[
                AccuracyMetricConfig
                | PrecisionMetricConfig
                | RecallMetricConfig
                | F1ScoreMetricConfig
                | ConfusionMatrixMetricConfig,
                Field(discriminator="name"),
            ]
        ]
        | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def _validate_transforms(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "train_transform" not in values or values["train_transform"] is None:
            values["train_transform"] = SequenceClassificationDocumentProcessor(
                hf_processor=HuggingfaceProcessor(
                    tokenizer_name="bert-base-uncased"  # default tokenizer
                ),
                overflow_strategy="return_first",
            )
        else:
            values["train_transform"] = (
                SequenceClassificationDocumentProcessor.model_validate(
                    values["train_transform"]
                )
            )
        if "eval_transform" not in values or values["eval_transform"] is None:
            values["eval_transform"] = SequenceClassificationDocumentProcessor(
                hf_processor=HuggingfaceProcessor(
                    tokenizer_name="bert-base-uncased"  # default tokenizer
                ),
                overflow_strategy="return_first",
            )
        else:
            values["eval_transform"] = (
                SequenceClassificationDocumentProcessor.model_validate(
                    values["eval_transform"]
                )
            )
        if "metrics" not in values or values["metrics"] is None:
            values["metrics"] = [
                AccuracyMetricConfig(),
                PrecisionMetricConfig(),
                RecallMetricConfig(),
                F1ScoreMetricConfig(),
                ConfusionMatrixMetricConfig(),
            ]
        return values


@MODEL_PIPELINES.register("sequence_classification")
class SequenceClassificationPipeline(SequenceModelPipeline):
    __config__ = SequenceClassificationPipelineConfig

    def _model_build_kwargs(self) -> dict[str, object]:
        assert self._labels.classification is not None, (
            "Labels must be provided for classification tasks."
            "Make sure the target dataset provides `labels.classification` in its metadata."
        )
        return {"num_labels": len(self._labels.classification)}

    def _prepare_labels(
        self, batch: DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        assert batch.label is not None, "Labels cannot be None"
        return {"labels": batch.label}

    def _get_logits(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, TransformersEncoderModelOutput):
            assert model_output.head_output is not None, "Head output cannot be None"
            assert isinstance(
                model_output.head_output, SequenceClassificationHeadOutput
            ), "Head output must be of type SequenceClassificationHeadOutput"
            assert model_output.head_output.logits is not None, "Logits cannot be None"
            return model_output.head_output.logits
        else:
            return model_output.logits

    def _compute_loss(
        self, model_output: Any, batch: DocumentTensorDataModel
    ) -> Tensor:
        from torch.nn.functional import cross_entropy

        assert batch.label is not None, "Labels cannot be None"
        loss = cross_entropy(self._get_logits(model_output=model_output), batch.label)
        return loss

    def _output_transform(
        self,
        loss: torch.Tensor | None,
        model_output: Any,
        batch: DocumentTensorDataModel,
    ) -> ModelOutput:
        assert self._labels.classification is not None, (
            "Labels must be provided for classification tasks."
        )
        assert batch.label is not None, "Labels cannot be None"
        logits = self._get_logits(model_output=model_output)
        predicted_labels = logits.argmax(dim=-1)
        return ClassificationModelOutput(
            loss=loss,
            logits=logits,
            prediction_probs=logits.softmax(dim=-1),
            gt_label_value=batch.label if batch.label is not None else None,
            gt_label_name=[self._labels.classification[i] for i in batch.label.tolist()]
            if batch.label is not None
            else None,
            predicted_label_value=predicted_labels
            if predicted_labels is not None
            else None,
            predicted_label_name=[
                self._labels.classification[i] for i in predicted_labels.tolist()
            ]
            if predicted_labels is not None
            else None,
        )

    def build_metrics(
        self,
        stage: Literal["train", "validation", "test"],
        device: torch.device | str = "cpu",
    ) -> dict[str, Metric]:
        if self.config.metrics is None:
            return {}
        assert self._labels.classification is not None, (
            "Labels must be provided for classification tasks."
        )
        metrics = {}
        for metric_config in self.config.metrics:
            logger.info(f"Building metric: {metric_config}")
            metric = metric_config.build(
                device=device, num_classes=len(self._labels.classification), stage=stage
            )
            metrics[metric_config.name] = metric
        return metrics


class TokenClassificationPipelineConfig(SequenceModelPipelineConfig):
    model: ModelConfig = ModelConfig(
        model_name_or_path="bert-base-uncased",
        builder_type=ModelBuilderType.transformers,
        model_type="token_classification",
    )
    name: str = "token_classification"
    metrics: (
        list[Annotated[SeqEvalMetricConfig, Field(discriminator="name")]] | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def _validate_transforms(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "train_transform" not in values or values["train_transform"] is None:
            values["train_transform"] = TokenClassificationDocumentProcessor(
                hf_processor=HuggingfaceProcessor(
                    tokenizer_name="bert-base-uncased"  # default tokenizer
                ),
                overflow_strategy="return_random",  # for token classification, we want all tokens
            )
        else:
            values["train_transform"] = (
                TokenClassificationDocumentProcessor.model_validate(
                    values["train_transform"]
                )
            )
        if "eval_transform" not in values or values["eval_transform"] is None:
            values["eval_transform"] = TokenClassificationDocumentProcessor(
                hf_processor=HuggingfaceProcessor(
                    tokenizer_name="bert-base-uncased"  # default tokenizer
                ),
                overflow_strategy="return_all",  # for token classification, we want all tokens
            )
        else:
            values["eval_transform"] = (
                TokenClassificationDocumentProcessor.model_validate(
                    values["eval_transform"]
                )
            )
        if "metrics" not in values or values["metrics"] is None:
            values["metrics"] = [SeqEvalMetricConfig()]
        return values


@MODEL_PIPELINES.register("token_classification")
class TokenClassificationPipeline(SequenceModelPipeline):
    __config__ = TokenClassificationPipelineConfig

    def _model_build_kwargs(self) -> dict[str, object]:
        assert self._labels.ser is not None, "Labels must be provided for ser tasks."
        return {"num_labels": len(self._labels.ser)}

    def _prepare_labels(
        self, batch: DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        assert batch.token_labels is not None, "Labels cannot be None"
        return {"labels": batch.token_labels}

    def _compute_loss(
        self, model_output: Any, batch: DocumentTensorDataModel
    ) -> Tensor:
        return model_output.loss

    def _output_transform(
        self,
        loss: torch.Tensor | None,
        model_output: Any,
        batch: DocumentTensorDataModel,
    ) -> ModelOutput:
        assert batch.token_labels is not None, "Token labels cannot be None"
        assert self._labels.ser is not None, "Labels must be provided for ser tasks."
        target_label_names = []
        predicted_label_names = []
        logits = model_output.logits
        predictions = logits.argmax(-1)
        for prediction, target in zip(predictions, batch.token_labels, strict=True):
            curr_target_label_names = [
                self._labels.ser[i] for i in target[target != -100]
            ]
            curr_predicted_label_names = [
                self._labels.ser[i] for i in prediction[target != -100]
            ]
            target_label_names.append(curr_target_label_names)
            predicted_label_names.append(curr_predicted_label_names)

        return TokenClassificationModelOutput(
            loss=loss,
            logits=logits,
            predicted_label_names=predicted_label_names,
            target_label_names=target_label_names,
        )

    def build_metrics(
        self,
        stage: Literal["train", "validation", "test"],
        device: torch.device | str = "cpu",
    ) -> dict[str, Metric]:
        if self.config.metrics is None:
            return {}
        metrics = {}
        for metric_config in self.config.metrics:
            logger.info(f"Building metric: {metric_config}")
            metric = metric_config.build(device=device, stage=stage)
            metrics[metric_config.name] = metric
        return metrics


class LayoutTokenClassificationPipelineConfig(SequenceModelPipelineConfig):
    model: ModelConfig = ModelConfig(
        model_name_or_path="bert-base-uncased",
        builder_type=ModelBuilderType.transformers,
        model_type="token_classification",
    )
    name: str = "layout_token_classification"
    metrics: (
        list[
            Annotated[
                LayoutPrecisionMetricConfig
                | LayoutRecallMetricConfig
                | LayoutF1MetricConfig
                | LayoutPrecisionMacroMetricConfig
                | LayoutRecallMacroMetricConfig
                | LayoutF1MacroMetricConfig,
                Field(discriminator="name"),
            ]
        ]
        | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def _validate_transforms(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "train_transform" not in values or values["train_transform"] is None:
            values["train_transform"] = TokenClassificationDocumentProcessor(
                hf_processor=HuggingfaceProcessor(
                    tokenizer_name="bert-base-uncased"  # default tokenizer
                ),
                overflow_strategy="return_random",  # for token classification, we want all tokens
            )
        else:
            values["train_transform"] = (
                TokenClassificationDocumentProcessor.model_validate(
                    values["train_transform"]
                )
            )
        if "eval_transform" not in values or values["eval_transform"] is None:
            values["eval_transform"] = TokenClassificationDocumentProcessor(
                hf_processor=HuggingfaceProcessor(
                    tokenizer_name="bert-base-uncased"  # default tokenizer
                ),
                overflow_strategy="return_all",  # for token classification, we want all tokens
            )
        else:
            values["eval_transform"] = (
                TokenClassificationDocumentProcessor.model_validate(
                    values["eval_transform"]
                )
            )
        if "metrics" not in values or values["metrics"] is None:
            values["metrics"] = [
                LayoutPrecisionMetricConfig(),
                LayoutRecallMetricConfig(),
                LayoutF1MetricConfig(),
                LayoutPrecisionMacroMetricConfig(average="macro"),
                LayoutRecallMacroMetricConfig(average="macro"),
                LayoutF1MacroMetricConfig(average="macro"),
            ]
        return values


@MODEL_PIPELINES.register("layout_token_classification")
class LayoutTokenClassificationPipeline(TokenClassificationPipeline):
    __config__ = LayoutTokenClassificationPipelineConfig

    def _input_transform(
        self, batch: DocumentTensorDataModel, require_labels: bool = False
    ) -> dict[str, torch.Tensor | None]:
        inputs = super()._input_transform(batch, require_labels=require_labels)
        assert self.config.use_bbox, (
            f"{self.__class__.__name__} requires use_bbox to be True in the config."
        )
        assert "bbox" in self._model_args_list, (
            f"{self._model.__class__.__name__} does not support 'bbox' as input argument and cannot be used "
            f"with {self.__class__.__name__}."
        )
        assert batch.token_bboxes is not None, (
            f"{self.__class__.__name__} requires token_bboxes in the batch. Found: {batch.token_bboxes=}"
        )
        assert "bbox" in inputs, (
            f"{self.__class__.__name__} requires 'bbox' in the model inputs"
        )
        return inputs

    def _output_transform(
        self,
        loss: torch.Tensor | None,
        model_output: Any,
        batch: DocumentTensorDataModel,
    ) -> ModelOutput:
        assert batch.token_labels is not None, "Token labels cannot be None"
        assert batch.token_bboxes is not None, (
            "For layout token classification, token bboxes cannot be None"
        )
        return LayoutTokenClassificationModelOutput(
            loss=loss,
            layout_token_logits=model_output.logits,
            layout_token_targets=batch.token_labels,
            layout_token_bboxes=batch.token_bboxes,
        )


class QuestionAnsweringPipelineConfig(SequenceModelPipelineConfig):
    model: ModelConfig = ModelConfig(
        model_name_or_path="bert-base-uncased",
        builder_type=ModelBuilderType.transformers,
        model_type="question_answering",
    )
    name: str = "question_answering"
    metrics: (
        list[Annotated[DueEvalMetricConfig, Field(discriminator="name")]] | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def _validate_transforms(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "train_transform" not in values or values["train_transform"] is None:
            values["train_transform"] = QuestionAnsweringDocumentProcessor(
                hf_processor=HuggingfaceProcessor(
                    tokenizer_name="bert-base-uncased"  # default tokenizer
                ),
                overflow_strategy="return_all",  # for QA, we want all tokens
            )
        else:
            values["train_transform"] = (
                QuestionAnsweringDocumentProcessor.model_validate(
                    values["train_transform"]
                )
            )
        if "eval_transform" not in values or values["eval_transform"] is None:
            values["eval_transform"] = QuestionAnsweringDocumentProcessor(
                hf_processor=HuggingfaceProcessor(
                    tokenizer_name="bert-base-uncased"  # default tokenizer
                ),
                overflow_strategy="return_all",  # for QA, we want all tokens
            )
        else:
            values["eval_transform"] = (
                QuestionAnsweringDocumentProcessor.model_validate(
                    values["eval_transform"]
                )
            )
        if "metrics" not in values or values["metrics"] is None:
            values["metrics"] = [DocVQAEvalConfig()]
        return values


@MODEL_PIPELINES.register("question_answering")
class QuestionAnsweringPipeline(SequenceModelPipeline):
    __config__ = QuestionAnsweringPipelineConfig

    def _model_build_kwargs(self) -> dict[str, object]:
        assert self._labels.ser is not None, "Labels must be provided for ser tasks."
        return {"num_labels": len(self._labels.ser)}

    def _prepare_labels(
        self, batch: DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        assert batch.token_answer_start is not None, "Labels cannot be None"
        assert batch.token_answer_end is not None, "Labels cannot be None"
        return {
            "start_positions": batch.token_answer_start,
            "end_positions": batch.token_answer_end,
        }

    def _compute_loss(
        self, model_output: Any, batch: DocumentTensorDataModel
    ) -> Tensor:
        return model_output.loss

    def _output_transform(
        self,
        loss: torch.Tensor | None,
        model_output: Any,
        batch: DocumentTensorDataModel,
    ) -> QAModelOutput:
        pred_answers_per_question_id = _postprocess_qa_predictions(
            words=batch.metadata.words,
            word_ids=batch.word_ids.detach().cpu(),
            sequence_ids=batch.sequence_ids.detach().cpu(),
            question_ids=batch.metadata.sample_id,  # each sample has a single question index for uniqueness
            start_logits=model_output.start_logits.detach().cpu(),
            end_logits=model_output.end_logits.detach().cpu(),
        )

        qa_outputs = []
        for (qid, preds), question in zip(
            pred_answers_per_question_id.items(),
            batch.metadata.qa_question,
            strict=True,
        ):
            if "_page_" in qid:
                sample_id = qid.split("_page_")[0]  # get the original sample id
            else:
                sample_id = qid.split("_subsample_")[0]  # get the original sample id
            answer = preds[0]["text"]  # taking the top prediction

            # we ignore samples with no answer during evaluation
            qa_outputs.append(
                QAPair(sample_id=sample_id, question=question, answer=answer)
            )

        return QAModelOutput(loss=loss, qa_pairs=qa_outputs)
