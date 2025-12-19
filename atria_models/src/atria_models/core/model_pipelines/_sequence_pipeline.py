from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from atria_logger import get_logger
from atria_transforms.data_types._document import DocumentTensorDataModel
from torch._tensor import Tensor

from atria_models.core.model_pipelines._common import ModelPipelineConfig
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from atria_models.core.model_pipelines.utilities import _postprocess_qa_predictions
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
        model_output = self._model(**inputs)
        loss = self._compute_loss(model_output=model_output, batch=batch)
        return self._output_transform(loss=loss, model_output=model_output, batch=batch)

    def evaluation_step(  # type: ignore[override]
        self,
        batch: DocumentTensorDataModel,
        stage: Literal["validation", "test"],
        **kwargs,
    ) -> ModelOutput:
        inputs = self._input_transform(batch, require_labels=True)
        model_output = self._model(**inputs)
        loss = self._compute_loss(model_output=model_output, batch=batch)
        return self._output_transform(loss=loss, model_output=model_output, batch=batch)

    def predict_step(  # type: ignore[override]
        self, batch: DocumentTensorDataModel, **kwargs
    ) -> ModelOutput:
        inputs = self._input_transform(batch, require_labels=False)
        model_output = self._model(**inputs)
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
        inputs = {
            "input_ids": batch.token_ids,
            "token_type_ids": batch.token_type_ids,
            "attention_mask": batch.attention_mask,
        }

        if "pixel_values" in self._model_args_list and self.config.use_image:
            assert batch.image is not None, "Image cannot be None"
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


@MODEL_PIPELINES.register("sequence_classification")
class SequenceClassificationPipeline(SequenceModelPipeline):
    __config__ = SequenceModelPipelineConfig

    def _model_build_kwargs(self) -> dict[str, object]:
        assert self._labels.classification is not None, (
            "Labels must be provided for classification tasks."
        )
        return {"num_labels": len(self._labels.classification)}

    def _prepare_labels(
        self, batch: DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        assert batch.label is not None, "Labels cannot be None"
        return {"labels": batch.label}

    def _compute_loss(
        self, model_output: Any, batch: DocumentTensorDataModel
    ) -> Tensor:
        from torch.nn.functional import cross_entropy

        assert batch.label is not None, "Labels cannot be None"
        logits = model_output.logits
        loss = cross_entropy(logits, batch.label)
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
        logits = model_output.logits
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


@MODEL_PIPELINES.register("token_classification")
class TokenClassificationPipeline(SequenceModelPipeline):
    __config__ = SequenceModelPipelineConfig

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


@MODEL_PIPELINES.register("layout_token_classification")
class LayoutTokenClassificationPipeline(TokenClassificationPipeline):
    def _output_transform(
        self,
        loss: torch.Tensor | None,
        model_output: Any,
        batch: DocumentTensorDataModel,
    ) -> ModelOutput:
        assert batch.token_labels is not None, "Token labels cannot be None"
        assert self._labels.ser is not None, "Labels must be provided for ser tasks."
        target_word_labels = batch.token_labels[batch.token_labels != -100]
        predicted_word_labels = model_output.logits.argmax(-1)[
            batch.token_labels != -100
        ]
        word_logits = model_output.logits[:, batch.token_labels != -100, :]
        return LayoutTokenClassificationModelOutput(
            loss=loss,
            word_logits=word_logits,
            predicted_word_labels=predicted_word_labels,
            target_word_labels=target_word_labels,
        )


@MODEL_PIPELINES.register("question_answering")
class QuestionAnsweringPipeline(SequenceModelPipeline):
    __config__ = SequenceModelPipelineConfig

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
