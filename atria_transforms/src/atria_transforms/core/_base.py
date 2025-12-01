from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import torch
from atria_logger import get_logger
from atria_types._utilities._repr import RepresentationMixin
from PIL.Image import Image as PILImage
from pydantic import BaseModel, ConfigDict

from atria_transforms.types._base import T_TensorDataModel

logger = get_logger(__name__)

T = TypeVar("T")


class DataTransform(RepresentationMixin, BaseModel, Generic[T_TensorDataModel]):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    @abstractmethod
    def __call__(self, instance: Any) -> T_TensorDataModel | list[T_TensorDataModel]:
        raise NotImplementedError


class ToRGB:
    def __call__(self, image: PILImage | torch.Tensor) -> PILImage | torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:
                return image
            return image.repeat(3, 1, 1)
        else:
            return image.convert("RGB")
