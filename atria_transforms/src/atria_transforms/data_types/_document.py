from __future__ import annotations

from collections.abc import Callable

import torch
from atria_types._generic._annotations import AnnotationType
from pydantic import BaseModel

from atria_transforms.core import TensorDataModel
from atria_transforms.data_types._tokenized_document_instance import (
    TokenizedDocumentInstance,
)


class DocumentTensorDataModel(TensorDataModel):
    class Metadata(BaseModel):
        index: int | None
        sample_id: str
        words: list[str]
        question_id: int | None = None
        qa_question: str | None = None
        qa_answers: list[str] | None = None
        bbox_normalized: bool = True

    token_ids: torch.Tensor
    word_ids: torch.Tensor
    sequence_ids: torch.Tensor
    token_bboxes: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    # sample level fields
    image: torch.Tensor | None = None
    label: torch.Tensor | None = None

    # extractive QA specific fields
    token_answer_start: torch.Tensor | None = None
    token_answer_end: torch.Tensor | None = None

    @classmethod
    def from_tokenized_instance(
        cls, tokenized_instance: TokenizedDocumentInstance, image_transform: Callable
    ) -> DocumentTensorDataModel:
        image_tensor = (
            image_transform(tokenized_instance.image.content)
            if tokenized_instance.image is not None
            else None
        )

        qa_metadata = {}
        if tokenized_instance.has_annotation_type(
            annotation_type=AnnotationType.question_answering
        ):
            annotation = tokenized_instance.get_annotation_by_type(
                annotation_type=AnnotationType.question_answering
            )
            qa_pairs = annotation.qa_pairs
            assert len(qa_pairs) == 1, (
                "Conversion from TokenizedDocumentInstance to DocumentTensorDataModel "
                "for question answering is only supported with a single qa_pair per tokenized_instance"
            )
            qa_metadata["question_id"] = qa_pairs[0].id
            qa_metadata["qa_question"] = qa_pairs[0].question_text
            qa_metadata["qa_answers"] = qa_pairs[0].answers
        return cls(
            index=tokenized_instance.index,
            sample_id=tokenized_instance.sample_id,
            words=tokenized_instance.words,
            token_ids=tokenized_instance.token_ids,
            word_ids=tokenized_instance.word_ids,
            sequence_ids=tokenized_instance.sequence_ids,
            token_bboxes=tokenized_instance.token_bboxes,
            token_type_ids=tokenized_instance.token_type_ids,
            token_labels=tokenized_instance.token_labels,
            attention_mask=tokenized_instance.attention_mask,
            image=image_tensor,
            label=tokenized_instance.label,
            token_answer_start=tokenized_instance.token_answer_start,
            token_answer_end=tokenized_instance.token_answer_end,
            **qa_metadata,
        )
