from __future__ import annotations

import numpy as np
from atria_types._generic._annotations import AnnotationType
from atria_types._generic._bounding_box import BoundingBox
from pydantic import BaseModel

from atria_transforms.core import TensorDataModel
from atria_transforms.data_types._tokenized_document_instance import (
    TokenizedDocumentInstance,
)

# np.set_printoptions(threshold=0, edgeitems=1, linewidth=80)


class DocumentTensorDataModel(TensorDataModel):
    class Metadata(BaseModel):
        index: int | None
        sample_id: str
        words: list[str]
        question_id: int | None = None
        qa_question: str | None = None
        qa_answers: list[str] | None = None
        bbox_normalized: bool = True
        is_embedding: bool = False

    token_ids: np.ndarray
    position_ids: np.ndarray | None = None
    word_ids: np.ndarray
    special_tokens_mask: np.ndarray | None = None
    sequence_ids: np.ndarray
    token_bboxes: np.ndarray | None = None
    layout_embeddings: np.ndarray | None = None
    token_type_ids: np.ndarray | None = None
    token_labels: np.ndarray | None = None
    attention_mask: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    segment_position_ids: np.ndarray | None = None
    valid_spans: np.ndarray | None = None

    image: np.ndarray | None = None
    label: np.ndarray | None = None

    token_answer_start: np.ndarray | None = None
    token_answer_end: np.ndarray | None = None

    @property
    def words(self) -> list[str]:
        return self.metadata.words

    @property
    def word_bboxes(self) -> list[BoundingBox]:
        word_bboxes = {}
        for word_idx in self.word_ids:
            if word_idx == -100:
                continue
            word_idx = int(word_idx)
            if word_idx not in word_bboxes:
                bbox = BoundingBox(value=self.token_bboxes[word_idx].tolist())
                word_bboxes[word_idx] = bbox

        word_bboxes = list(word_bboxes.values())
        assert len(word_bboxes) == len(self.words), (
            "Number of word bounding boxes does not match number of words"
        )
        return word_bboxes

    @classmethod
    def from_tokenized_instance(
        cls, tokenized_instance: TokenizedDocumentInstance
    ) -> DocumentTensorDataModel:
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
                "for question answering is only supported with a single qa_pair"
            )
            qa_metadata["question_id"] = qa_pairs[0].id
            qa_metadata["qa_question"] = qa_pairs[0].question_text
            qa_metadata["qa_answers"] = qa_pairs[0].answers

        image = None
        if tokenized_instance.image is not None:
            assert isinstance(tokenized_instance.image, np.ndarray), (
                "Image content must be a numpy array for conversion to DocumentTensorDataModel"
            )
            image = tokenized_instance.image

        return cls(
            index=tokenized_instance.index,
            sample_id=tokenized_instance.sample_id,
            words=tokenized_instance.words,
            token_ids=tokenized_instance.token_ids,
            word_ids=tokenized_instance.word_ids,
            special_tokens_mask=tokenized_instance.special_tokens_mask,
            sequence_ids=tokenized_instance.sequence_ids,
            token_bboxes=tokenized_instance.token_bboxes,
            token_type_ids=tokenized_instance.token_type_ids,
            token_labels=tokenized_instance.token_labels,
            attention_mask=tokenized_instance.attention_mask,
            image=image,
            segment_ids=tokenized_instance.segment_ids,
            segment_position_ids=tokenized_instance.segment_position_ids,
            position_ids=tokenized_instance.segment_position_ids,
            valid_spans=tokenized_instance.valid_spans,
            label=tokenized_instance.label,
            token_answer_start=tokenized_instance.token_answer_start,
            token_answer_end=tokenized_instance.token_answer_end,
            **qa_metadata,
        )
