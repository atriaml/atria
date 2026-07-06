from __future__ import annotations

from typing import Any

import numpy as np
from atria_logger import get_logger
from atria_types._data_instance._base import BaseDataInstance
from atria_types._generic._image import Image
from pydantic import field_serializer, field_validator

logger = get_logger(__name__)


class TokenizedDocumentInstance(BaseDataInstance):
    image: Image | None = None
    words: list[str]
    token_ids: np.ndarray
    word_ids: np.ndarray
    special_tokens_mask: np.ndarray | None = None
    sequence_ids: np.ndarray
    token_bboxes: np.ndarray | None = None
    token_type_ids: np.ndarray | None = None
    token_labels: np.ndarray | None = None
    attention_mask: np.ndarray | None = None
    label: np.ndarray | None = None
    token_answer_start: np.ndarray | None = None
    token_answer_end: np.ndarray | None = None
    overflow_resolved: bool = False

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @field_serializer(
        "token_ids",
        "word_ids",
        "special_tokens_mask",
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
    def serialize_array(cls, value: np.ndarray | None) -> list | None:
        if value is None:
            return None
        return value.tolist()

    @field_validator(
        "token_ids",
        "word_ids",
        "special_tokens_mask",
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
    def validate_array(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, list):
            return np.array(value)
        else:
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    return value.numpy()
            except ImportError:
                pass
            raise ValueError(f"Unsupported type for array field: {type(value)}")

    def resolve_overflow(
        self, overflow_idx: int, update_sample_id: bool = False
    ) -> TokenizedDocumentInstance:
        batch_size = self.token_ids.shape[0]

        def _get_at_idx(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            assert len(arr) == batch_size, (
                f"Array batch size {len(arr)} does not match expected "
                f"batch size {batch_size}"
            )
            if arr.ndim == 1:
                return np.array(arr[overflow_idx])
            return arr[overflow_idx]

        kwargs = {}
        if update_sample_id:
            kwargs = {"sample_id": f"{self.sample_id}_overflow_{overflow_idx}"}

        return self.model_copy(
            update={
                **kwargs,
                "token_ids": _get_at_idx(self.token_ids),
                "word_ids": _get_at_idx(self.word_ids),
                "special_tokens_mask": _get_at_idx(self.special_tokens_mask),
                "sequence_ids": _get_at_idx(self.sequence_ids),
                "token_bboxes": _get_at_idx(self.token_bboxes),
                "token_type_ids": _get_at_idx(self.token_type_ids),
                "token_labels": _get_at_idx(self.token_labels),
                "attention_mask": _get_at_idx(self.attention_mask),
                "label": _get_at_idx(self.label),
                "token_answer_start": _get_at_idx(self.token_answer_start),
                "token_answer_end": _get_at_idx(self.token_answer_end),
                "overflow_resolved": True,
            }
        )
