from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from atria_logger import get_logger
from atria_types import DocumentInstance
from atria_types._generic._annotations import (
    AnnotationType,
    QuestionAnsweringAnnotation,
)
from atria_types._generic._qa_pair import QAPair

from atria_transforms.core._tfs._base import DataTransform
from atria_transforms.data_types._document import DocumentTensorDataModel

from ...registry import DATA_TRANSFORMS
from .._utilities import (
    _document_instance_to_hf_processor_inputs,
    _generate_qa_token_ids,
)
from ._base import DocumentProcessor

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
        import torch

        processed_outputs = super()._post_process_tokenizer_outputs(
            document_instance, hf_processor_inputs, tokenization_data
        )

        label = document_instance.get_annotation_by_type(
            AnnotationType.classification
        ).label.value
        processed_outputs["label"] = torch.tensor(label, dtype=torch.long)
        return processed_outputs

    def _processed_outputs_from_tokenized(
        self, document_instance: PreTokenizedDocumentInstance
    ) -> dict[str, Any]:
        processed_outputs = super()._processed_outputs_from_tokenized(document_instance)
        label = document_instance.get_annotation_by_type(
            AnnotationType.classification
        ).label.value
        processed_outputs["label"] = torch.tensor(label, dtype=torch.long)
        return processed_outputs


@DATA_TRANSFORMS.register("token_classification_document_processor")
class TokenClassificationDocumentProcessor(DocumentProcessor):
    pass


@DATA_TRANSFORMS.register("unroll_qa_pairs_transform")
class UnrollQAPairsTransform(DataTransform[list[DocumentInstance]]):
    remove_no_answer_samples: bool = False

    def __call__(self, document_instance: DocumentInstance) -> list[DocumentInstance]:
        qa_annotations = document_instance.get_annotation_by_type(
            annotation_type=AnnotationType.question_answering
        )

        document_instance_per_qa_pair = []
        for qa_pair in qa_annotations.qa_pairs:
            annotation = QuestionAnsweringAnnotation(qa_pairs=[qa_pair])

            def has_answer(qa_pair: QAPair):
                for answer_span in qa_pair.answer_spans:
                    if answer_span.start != -1 and answer_span.end != -1:
                        return True
                return False

            assert qa_pair.answer_spans is not None
            if self.remove_no_answer_samples:
                if not has_answer(qa_pair):
                    logger.debug(
                        f"Skipping QA Pair with id {qa_pair.id} due to no answer spans."
                    )
                    continue

            new_document_instance = document_instance.model_copy(
                update={
                    "sample_id": f"{document_instance.sample_id}_qa_{qa_pair.id}",
                    "annotations": [annotation],
                }
            )
            document_instance_per_qa_pair.append(new_document_instance)

        return document_instance_per_qa_pair


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
            tokenization_data = self.hf_processor(hf_processor_inputs)

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
        import numpy as np

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

    def _resolve_overflow(
        self, processed_outputs: dict[str, Any], overflow_sample_idx: int
    ) -> DocumentTensorDataModel:
        data = {
            key: value[overflow_sample_idx] if value is not None else None
            for key, value in processed_outputs.items()
            if key
            in [
                "token_ids",
                "attention_mask",
                "token_bboxes",
                "token_type_ids",
                "token_labels",
                "sequence_ids",
                "word_ids",
                "token_answer_start",
                "token_answer_end",
            ]
        }

        # image is the same for all overflowing segments of the same document
        if "image":
            data["image"] = processed_outputs["image"]

        # if label is present at sample level, add it
        if "label" in processed_outputs:
            data["label"] = processed_outputs["label"]

        # by default, we return for eery overflowing segment a separate DocumentTensorDataModel
        index = processed_outputs.get("index", None)
        assert index is not None, "index must be present in processed outputs"
        sample_id = processed_outputs.get("sample_id", None)
        assert sample_id is not None, "sample_id must be present in processed outputs"
        words = processed_outputs.get("words", None)
        assert words is not None, "words must be present in processed outputs"
        return DocumentTensorDataModel(
            index=index,
            sample_id=sample_id,
            words=words,
            question_id=processed_outputs.get("question_id", None),
            qa_question=processed_outputs.get("qa_question", None),
            qa_answers=processed_outputs.get("qa_answer", None),
            **data,
        )
