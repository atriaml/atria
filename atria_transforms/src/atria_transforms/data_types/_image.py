from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from atria_transforms.core import TensorDataModel

if TYPE_CHECKING:
    import torch


class ImageTensorDataModel(TensorDataModel):
    class Metadata(BaseModel):
        index: int | None
        sample_id: str
        words: list[str]
        question_id: int | None = None
        qa_question: str | None = None
        qa_answers: list[str] | None = None

    # sample level fields
    image: torch.Tensor
    label: torch.Tensor | None = None

    # extractive QA specific fields
    token_answer_start: torch.Tensor | None = None
    token_answer_end: torch.Tensor | None = None
