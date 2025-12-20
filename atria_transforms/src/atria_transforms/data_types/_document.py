from __future__ import annotations

import torch
from pydantic import BaseModel

from atria_transforms.core import TensorDataModel


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
    overflow_to_sample_mapping: torch.Tensor | None = None

    # segment level fields
    segment_index: torch.Tensor | None = None
    segment_inner_token_rank: torch.Tensor | None = None
    first_token_idxes: torch.Tensor | None = None
    first_token_idxes_mask: torch.Tensor | None = None

    # sample level fields
    image: torch.Tensor | None = None
    label: torch.Tensor | None = None

    # extractive QA specific fields
    token_answer_start: torch.Tensor | None = None
    token_answer_end: torch.Tensor | None = None
