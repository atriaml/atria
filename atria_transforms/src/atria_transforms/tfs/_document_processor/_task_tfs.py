from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_types import DocumentInstance
from atria_types._generic._annotations import AnnotationType
from atria_types._generic._qa_pair import QAPair

from atria_transforms.data_types._document import DocumentTensorDataModel

from ...registry import DATA_TRANSFORMS
from ._base import DocumentProcessor
from ._utilities import (
    _document_instance_to_hf_processor_inputs,
    _generate_qa_token_ids,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import BatchEncoding


logger = get_logger(__name__)


@DATA_TRANSFORMS.register("sequence_classification_document_processor")
class SequenceClassificationDocumentProcessor(DocumentProcessor):
    def _post_process_tokenizer_outputs(
        self,
        document_instance: DocumentInstance,
        hf_processor_inputs: dict[str, Any],
        tokenization_data: BatchEncoding,
    ) -> dict[str, Any]:
        processed_outputs = super()._post_process_tokenizer_outputs(
            document_instance, hf_processor_inputs, tokenization_data
        )

        processed_outputs["label"] = document_instance.get_annotation_by_type(
            AnnotationType.classification
        ).label.value
        return processed_outputs


@DATA_TRANSFORMS.register("token_classification_document_processor")
class TokenClassificationDocumentProcessor(DocumentProcessor):
    pass


@DATA_TRANSFORMS.register("question_answering_document_processor")
class QuestionAnsweringDocumentProcessor(DocumentProcessor):
    ignore_samples_with_no_answer: bool = False
    is_training: bool = False
    truncation: str = "only_second"

    def _is_no_answer_sample(
        self, token_answer_start, token_answer_end, tokenization_data
    ):
        import numpy as np

        total_answers = len(token_answer_start)
        for key, value in tokenization_data.items():
            if value is None:
                continue
            if key == "image":
                continue
            assert len(value) == total_answers, (
                f"Length mismatch in tokenization data for key {key}. "
                f"Expected length: {total_answers}, Actual length: {len(value)}"
            )

        valid_indices = []
        for idx, (s, e) in enumerate(
            zip(token_answer_start, token_answer_end, strict=True)
        ):
            if s != -1 and e != -1:
                valid_indices.append(idx)

        if len(valid_indices) == 0:
            return True  # skip this sample entirely

        if len(valid_indices) < total_answers:
            tokenization_data = {
                k: v[valid_indices] if v is not None and k not in ["image"] else v
                for k, v in tokenization_data.items()
            }
            token_answer_start = token_answer_start[valid_indices]
            token_answer_end = token_answer_end[valid_indices]

        assert (np.array(token_answer_end) != -1).all(), (
            f"Some end answer indices are -1 in document {token_answer_end}"
        )
        assert (np.array(token_answer_start) != -1).all(), (
            f"Some start answer indices are -1 in document {token_answer_start}"
        )
        total_answers = len(token_answer_start)
        for key, value in tokenization_data.items():
            if value is None:
                continue
            if key == "image":
                continue
            assert len(value) == total_answers, (
                f"Length mismatch in tokenization data for key {key}. "
                f"Expected length: {total_answers}, Actual length: {len(value)}"
            )
        return False

    def __call__(
        self, document_instance: DocumentInstance
    ) -> DocumentTensorDataModel | list[DocumentTensorDataModel]:
        qa_pairs = document_instance.get_annotation_by_type(
            AnnotationType.question_answering
        ).qa_pairs
        assert qa_pairs is not None, "No QA pairs found in the document instance."
        assert len(qa_pairs) > 0, "No QA pairs found in the document instance."

        transformed_instances = []
        for qa_pair_index in range(len(qa_pairs)):
            # convert DocumentInstance to Huggingface processor inputs
            hf_processor_inputs = _document_instance_to_hf_processor_inputs(
                document_instance,
                use_segment_level_bboxes=self.use_segment_level_bboxes,
                image_transform=self.image_transform,
                context=qa_pairs[qa_pair_index].question_text,
            )

            # perform tokenization using the hf_processor
            tokenization_data = self.hf_processor(**hf_processor_inputs)

            # post-process tokenizer outputs to generate segment-level info and align word ids and labels
            processed_outputs = self._post_process_qa_tokenizer_outputs(
                document_instance=document_instance,
                hf_processor_inputs=hf_processor_inputs,
                tokenization_data=tokenization_data,
                qa_pair=qa_pairs[qa_pair_index],
            )

            # for each token processed_outputs, create DocumentTensorDataModel
            bs = processed_outputs["token_ids"].shape[0]
            processed = [
                self._resolve_overflow(
                    processed_outputs=processed_outputs,
                    overflow_sample_idx=overflow_sample_idx,
                )
                for overflow_sample_idx in range(bs)
            ]
            if bs == 1:
                return processed[0]
            return processed

        return transformed_instances

    def _post_process_qa_tokenizer_outputs(
        self,
        document_instance: DocumentInstance,
        hf_processor_inputs: dict[str, Any],
        tokenization_data: BatchEncoding,
        qa_pair: QAPair,
    ) -> dict[str, Any]:
        processed_outputs = self._post_process_tokenizer_outputs(
            document_instance, hf_processor_inputs, tokenization_data
        )

        token_answer_start, token_answer_end = _generate_qa_token_ids(
            qa_pair=qa_pair,
            word_ids=processed_outputs["word_ids"],
            sequence_ids=processed_outputs["sequence_ids"],
            sequence_length=processed_outputs["token_ids"].shape[-1],
        )

        # if all token_answer_start and token_answer_end are 0, it means we could not find the answer in the context
        # therefore using this sample as a training sample will not help the model learn anything
        if self.is_training and self.ignore_samples_with_no_answer:
            total_answers = len(token_answer_start)
            for key, value in processed_outputs.items():
                if value is None:
                    continue
                if key == "image":
                    continue
                assert len(value) == total_answers, (
                    f"Length mismatch in tokenization data for key {key}. "
                    f"Expected length: {total_answers}, Actual length: {len(value)}"
                )

            valid_indices = []
            for idx, (s, e) in enumerate(
                zip(token_answer_start, token_answer_end, strict=True)
            ):
                if s != -1 and e != -1:
                    valid_indices.append(idx)

            if len(valid_indices) == 0:
                raise ValueError(
                    "All QA pairs in this document instance have no answer. "
                    "Consider setting ignore_samples_with_no_answer to False."
                )

            if len(valid_indices) < total_answers:
                processed_outputs = {
                    k: v[valid_indices] if v is not None and k not in ["image"] else v
                    for k, v in processed_outputs.items()
                }
                token_answer_start = token_answer_start[valid_indices]
                token_answer_end = token_answer_end[valid_indices]

            assert (np.array(token_answer_end) != -1).all(), (
                f"Some end answer indices are -1 in document {token_answer_end}"
            )
            assert (np.array(token_answer_start) != -1).all(), (
                f"Some start answer indices are -1 in document {token_answer_start}"
            )
            total_answers = len(token_answer_start)
            for key, value in processed_outputs.items():
                if value is None:
                    continue
                if key == "image":
                    continue
                assert len(value) == total_answers, (
                    f"Length mismatch in tokenization data for key {key}. "
                    f"Expected length: {total_answers}, Actual length: {len(value)}"
                )

        return {
            **processed_outputs,
            "token_answer_start": token_answer_start,
            "token_answer_end": token_answer_end,
            "question_id": qa_pair.id,
            "qa_question": qa_pair.question_text,
            "qa_answer": qa_pair.answers,
        }
