from __future__ import annotations

from typing import Any

import numpy as np
import torch
from atria_logger import get_logger
from atria_types._data_instance._base import BaseDataInstance
from atria_types._generic._image import Image
from pydantic import field_serializer, field_validator

logger = get_logger(__name__)


class TokenizedDocumentInstance(BaseDataInstance):
    image: Image | None = None
    words: list[str]
    token_ids: torch.Tensor
    word_ids: torch.Tensor
    sequence_ids: torch.Tensor
    token_bboxes: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    label: torch.Tensor | None = None
    token_answer_start: torch.Tensor | None = None
    token_answer_end: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @field_serializer(
        "token_ids",
        "word_ids",
        "sequence_ids",
        "token_bboxes",
        "token_type_ids",
        "token_labels",
        "label",
        "attention_mask",
        "token_answer_start",
        "token_answer_end",
        mode="plain",
    )
    @classmethod
    def serialize_tensor(cls, tensor: torch.Tensor) -> np.ndarray | None:
        if tensor is None:
            return tensor
        return tensor.numpy()

    @field_validator(
        "token_ids",
        "word_ids",
        "sequence_ids",
        "token_bboxes",
        "token_type_ids",
        "token_labels",
        "label",
        "attention_mask",
        "token_answer_start",
        "token_answer_end",
        mode="before",
    )
    @classmethod
    def validate_tensor(cls, value: Any) -> torch.Tensor:
        if value is None:
            return value
        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported type for tensor field: {type(value)}")

    def resolve_overflow(self, overflow_idx: int):
        batch_size = self.token_ids.shape[0]

        def _get_at_idx(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            assert len(tensor) == batch_size, (
                f"Tensor batch size {len(tensor)} does not match expected "
                f"batch size {batch_size}"
            )
            return tensor[overflow_idx]

        return self.model_copy(
            update={
                "token_ids": _get_at_idx(self.token_ids),
                "word_ids": _get_at_idx(self.word_ids),
                "sequence_ids": _get_at_idx(self.sequence_ids),
                "token_bboxes": _get_at_idx(self.token_bboxes),
                "token_type_ids": _get_at_idx(self.token_type_ids),
                "token_labels": _get_at_idx(self.token_labels),
                "attention_mask": _get_at_idx(self.attention_mask),
                "label": _get_at_idx(self.label),
                "token_answer_start": _get_at_idx(self.token_answer_start),
                "token_answer_end": _get_at_idx(self.token_answer_end),
            }
        )
