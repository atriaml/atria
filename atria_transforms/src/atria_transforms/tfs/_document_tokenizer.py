from __future__ import annotations

from typing import Self

import torch
from atria_logger import get_logger
from atria_types import DocumentInstance
from atria_types._generic._annotations import AnnotationType
from atria_types._generic._doc_content import DocumentContent
from atria_types._generic._image import Image
from atria_types._generic._qa_pair import QAPair
from pydantic import Field, model_validator

from atria_transforms.core import DataTransform
from atria_transforms.data_types._tokenized_document_instance import (
    TokenizedDocumentInstance,
)
from atria_transforms.registry import DATA_TRANSFORMS
from atria_transforms.tfs import HuggingfaceProcessor
from atria_transforms.tfs._hf_processor import (
    HuggingfaceProcessorInput,
    HuggingfaceProcessorOutput,
)
from atria_transforms.tfs._utilities import _generate_qa_token_ids

logger = get_logger(__name__)


class QAHuggingfaceProcessorOutput(HuggingfaceProcessorOutput):
    token_answer_start: torch.Tensor
    token_answer_end: torch.Tensor

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        for field_name in self.__class__.model_fields.keys():
            field_value = getattr(self, field_name)
            if field_value is not None:
                assert field_value.shape[0] == self.batch_size, (
                    f"Batch size mismatch for field '{field_name}': "
                    f"expected {self.batch_size}, got {field_value.shape[0]}"
                )
                if field_name not in ["token_answer_start", "token_answer_end"]:
                    assert field_value.shape[1] == self.sequence_length, (
                        f"Batch size mismatch for field '{field_name}': "
                        f"expected {self.batch_size}, got {field_value.shape[0]}"
                    )

        return self


class DocumentTokenizer(DataTransform[TokenizedDocumentInstance]):
    hf_processor: HuggingfaceProcessor = Field(default_factory=HuggingfaceProcessor)

    # segment-level-rank info args
    use_segment_level_bboxes: bool = False
    resize_image: tuple[int, int] | None = None
    load_image: bool = True
    load_bboxes: bool = True

    @property
    def data_model(self):
        return TokenizedDocumentInstance

    def _is_bboxes_normalized(self, content: DocumentContent) -> list[bool] | None:
        # add bboxes
        if self.load_bboxes:
            if self.use_segment_level_bboxes and len(content.segment_bbox_list) > 0:
                word_bboxes = content.segment_bbox_list
            else:
                word_bboxes = content.bbox_list
            return [bbox.normalized for bbox in word_bboxes]
        return None

    def _load_word_bboxes(self, content: DocumentContent) -> list[list[float]] | None:
        # add bboxes
        if self.load_bboxes:
            if self.use_segment_level_bboxes and len(content.segment_bbox_list) > 0:
                word_bboxes = content.segment_bbox_list
            else:
                word_bboxes = content.bbox_list
            word_bboxes = [bbox.value for bbox in word_bboxes]
            return word_bboxes
        return None

    def _get_classification_annotation_label(
        self, document_instance: DocumentInstance
    ) -> int | None:
        if document_instance.has_annotation_type(AnnotationType.classification):
            return document_instance.get_annotation_by_type(
                AnnotationType.classification
            ).label.value
        return None

    def _get_token_classification_labels(
        self, document_instance: DocumentInstance
    ) -> list[int] | None:
        if document_instance.has_annotation_type(AnnotationType.entity_labeling):
            entity_labeling_ann = document_instance.get_annotation_by_type(
                AnnotationType.entity_labeling
            )
            if entity_labeling_ann.word_labels is not None:
                return [label.value for label in entity_labeling_ann.word_labels]
        return None

    def _prepare_hf_processor_inputs(
        self, document_instance: DocumentInstance
    ) -> HuggingfaceProcessorInput:
        assert document_instance.content is not None, (
            f"{self.__class__.__name__} requires DocumentInstance to have content."
        )
        text = document_instance.content.text_list
        boxes = self._load_word_bboxes(document_instance.content)
        label = self._get_classification_annotation_label(document_instance)
        word_labels = self._get_token_classification_labels(document_instance)
        return HuggingfaceProcessorInput(
            text=text, boxes=boxes, label=label, word_labels=word_labels
        )

    def _prepare_image(self, document_instance: DocumentInstance) -> Image | None:
        if not self.load_image:
            return
        if document_instance.image is not None:
            if self.resize_image is not None:
                return document_instance.image.ops.to_rgb().ops.resize(
                    width=self.resize_image[0], height=self.resize_image[1]
                )
        return document_instance.image

    def _tokenize_instance(
        self, document_instance: DocumentInstance
    ) -> HuggingfaceProcessorOutput:
        # convert DocumentInstance to Huggingface processor inputs
        hf_processor_input = self._prepare_hf_processor_inputs(document_instance)

        # perform tokenization using the hf_processor
        return self.hf_processor(hf_processor_input)

    def __call__(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance | list[TokenizedDocumentInstance]:
        assert document_instance.content is not None, (
            f"{self.__class__.__name__} requires DocumentInstance to have content."
        )
        hf_processor_output = self._tokenize_instance(document_instance)

        tokenized_instance = TokenizedDocumentInstance(
            index=document_instance.index,
            sample_id=document_instance.sample_id,
            annotations=document_instance.annotations,
            image=self._prepare_image(document_instance),
            words=document_instance.content.text_list,
            token_ids=hf_processor_output.token_ids,
            word_ids=hf_processor_output.word_ids,
            sequence_ids=hf_processor_output.sequence_ids,
            token_bboxes=hf_processor_output.token_bboxes,
            token_type_ids=hf_processor_output.token_type_ids,
            token_labels=hf_processor_output.token_labels,
            attention_mask=hf_processor_output.attention_mask,
            label=hf_processor_output.label,
        )

        # for each token processed_outputs, create DocumentTensorDataModel
        return tokenized_instance


@DATA_TRANSFORMS.register("document_tokenizer/sequence_classification")
class SequenceClassificationDocumentTokenizer(DocumentTokenizer):
    def __call__(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance:
        instance = super().__call__(document_instance)
        assert instance.label is not None, (
            f"{self.__class__.__name__} requires the DocumentInstance "
            "to have a classification annotation with a label."
        )
        return instance


@DATA_TRANSFORMS.register("document_tokenizer/token_classification")
class TokenClassificationDocumentTokenizer(DocumentTokenizer):
    def __call__(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance | list[TokenizedDocumentInstance]:
        instance = super().__call__(document_instance)
        assert instance.token_labels is not None, (
            f"{self.__class__.__name__} requires the DocumentInstance "
            "to have a classification annotation with a token_labels."
        )
        return instance


@DATA_TRANSFORMS.register("document_tokenizer/question_answering")
class QuestionAnsweringDocumentTokenizer(DocumentTokenizer):
    hf_processor: HuggingfaceProcessor = Field(default_factory=HuggingfaceProcessor)
    ignore_no_answer_qa_pair: bool = False

    @property
    def data_model(self):
        return TokenizedDocumentInstance

    def _get_qa_pair(self, document_instance: DocumentInstance) -> QAPair:
        qa_pairs = document_instance.ops.get_qa_pairs()
        assert len(qa_pairs) == 1, (
            "DocumentInstance passed to _process_instance_for_qa_pair "
            "must contain only one QAPair in its QuestionAnsweringAnnotation."
        )
        assert len(qa_pairs[0].answer_spans) > 0, (
            f"QA Pair {qa_pairs[0].id} has no answer spans."
        )
        return qa_pairs[0]

    def _prepare_hf_processor_inputs(
        self, document_instance: DocumentInstance
    ) -> HuggingfaceProcessorInput:
        assert document_instance.content is not None, (
            f"{self.__class__.__name__} requires DocumentInstance to have content."
        )
        qa_pair = self._get_qa_pair(document_instance)
        text = qa_pair.question_text
        text_pair = document_instance.content.text_list
        boxes = self._load_word_bboxes(document_instance.content)
        label = self._get_classification_annotation_label(document_instance)
        word_labels = self._get_token_classification_labels(document_instance)
        if boxes is not None:
            assert len(text_pair) == len(boxes), (
                f"Length mismatch between text_pair and boxes for sample {document_instance.sample_id}. "
                f"Length of text_pair: {len(text_pair)}, Length of boxes: {len(boxes)}"
            )
        return HuggingfaceProcessorInput(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            label=label,
            word_labels=word_labels,
        )

    def _extract_token_answer_spans(
        self,
        document_instance: DocumentInstance,
        hf_processor_output: HuggingfaceProcessorOutput,
    ) -> QAHuggingfaceProcessorOutput:
        # first we get qa pair
        qa_pair = self._get_qa_pair(document_instance)

        # we generate token-level answer start and end positions
        # note each question can have multiple answers (hence multiple start/end positions)
        # each input can also be overflown so for each overflow instance, we have a separate start/end position
        # which corresponds to the tokenized input batch size
        # if overflow = 2, batch_size = 2, and we get 1 start/end position per overflow so
        # len(token_answer_start) = 2 and len(token_answer_end) = 2
        token_answer_start, token_answer_end = _generate_qa_token_ids(
            qa_pair=qa_pair,
            word_ids=hf_processor_output.word_ids,
            sequence_ids=hf_processor_output.sequence_ids,
            sequence_length=hf_processor_output.sequence_length,
        )
        total_answers = len(token_answer_start)
        assert hf_processor_output.batch_size == total_answers, (
            f"Batch size mismatch: expected {total_answers}, got {hf_processor_output.batch_size}"
        )

        if self.ignore_no_answer_qa_pair:
            # if we are ignoring no answer qa pairs, we filter out those with -1 start/end positions
            valid_indices = []
            for idx, (s, e) in enumerate(
                zip(token_answer_start, token_answer_end, strict=True)
            ):
                if s != -1 and e != -1:
                    valid_indices.append(idx)

            if len(valid_indices) == 0:
                raise ValueError(
                    "All QA pairs in this document instance have no answer. "
                    "Consider setting ignore_no_answer_qa_pair to False."
                )

            if len(valid_indices) < total_answers:
                token_answer_start = token_answer_start[valid_indices]
                token_answer_end = token_answer_end[valid_indices]
                hf_processor_output = hf_processor_output.take(valid_indices)
        return QAHuggingfaceProcessorOutput(
            **hf_processor_output.model_dump(),
            token_answer_start=token_answer_start,
            token_answer_end=token_answer_end,
        )

    def _tokenize_instance(
        self, document_instance: DocumentInstance
    ) -> QAHuggingfaceProcessorOutput:
        hf_processor_output = super()._tokenize_instance(document_instance)

        return self._extract_token_answer_spans(
            document_instance=document_instance, hf_processor_output=hf_processor_output
        )

    def _process_instance_for_qa_pair(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance | None:
        assert document_instance.content is not None, (
            f"{self.__class__.__name__} requires DocumentInstance to have content."
        )
        hf_processor_output = self._tokenize_instance(document_instance)

        tokenized_instance = TokenizedDocumentInstance(
            index=document_instance.index,
            sample_id=document_instance.sample_id,
            annotations=document_instance.annotations,
            image=self._prepare_image(document_instance),
            words=document_instance.content.text_list,
            token_ids=hf_processor_output.token_ids,
            word_ids=hf_processor_output.word_ids,
            sequence_ids=hf_processor_output.sequence_ids,
            token_bboxes=hf_processor_output.token_bboxes,
            token_type_ids=hf_processor_output.token_type_ids,
            token_labels=hf_processor_output.token_labels,
            attention_mask=hf_processor_output.attention_mask,
            token_answer_start=hf_processor_output.token_answer_start,
            token_answer_end=hf_processor_output.token_answer_end,
        )
        return tokenized_instance

    def __call__(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance | list[TokenizedDocumentInstance]:
        tokenized_qa_document_instances = []
        for (
            qa_pair_document_instance
        ) in document_instance.ops.get_instances_per_qa_pair(
            ignore_no_answer_qa_pair=self.ignore_no_answer_qa_pair
        ):
            tokenized_qa_document_instance = self._process_instance_for_qa_pair(
                qa_pair_document_instance
            )
            if tokenized_qa_document_instance is not None:
                if isinstance(tokenized_qa_document_instance, list):
                    tokenized_qa_document_instances.extend(
                        tokenized_qa_document_instance
                    )
                else:
                    tokenized_qa_document_instances.append(
                        tokenized_qa_document_instance
                    )
        return tokenized_qa_document_instances
