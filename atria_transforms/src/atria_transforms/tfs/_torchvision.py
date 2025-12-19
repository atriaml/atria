from enum import Enum
from typing import TYPE_CHECKING

from pydantic import ConfigDict

from atria_transforms.core._tfs._base import DataTransform
from atria_transforms.registry.registry_groups import DATA_TRANSFORMS

if TYPE_CHECKING:
    import torch


class InterpolationMode(str, Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


@DATA_TRANSFORMS.register("resize")
class ResizeTransform(DataTransform):  # or inherit from DataTransform if needed
    model_config = ConfigDict(extra="allow")
    size: list[int]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR
    antialias: bool = True

    def __call__(self, input: "torch.Tensor") -> "torch.Tensor":
        from torchvision.transforms.functional import (
            InterpolationMode as TVInterpolationMode,
            resize,
        )

        return resize(
            input,
            list(self.size),
            interpolation=TVInterpolationMode(self.interpolation.value),
            antialias=self.antialias,
        )
