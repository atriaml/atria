from __future__ import annotations

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

    @field_serializer(
        "token_ids",
        "word_ids",
        "sequence_ids",
        "token_bboxes",
        "token_type_ids",
        "token_labels",
        "attention_mask",
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
        "attention_mask",
        mode="before",
    )
    @classmethod
    def validate_tensor(
        cls, value: dict[str, int] | str | list[int] | torch.Tensor
    ) -> torch.Tensor:
        if value is None:
            return value
        if isinstance(value, torch.Tensor):
            return value

        return torch.from_numpy(value.copy())
