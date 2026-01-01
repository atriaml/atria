from __future__ import annotations

from typing import Literal

import torch
from atria_logger import get_logger
from PIL.Image import Image as PILImage

from atria_transforms.core import DataTransform
from atria_transforms.registry import DATA_TRANSFORMS

logger = get_logger(__name__)

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class ToRGB(DataTransform[PILImage]):
    def __call__(self, image: PILImage) -> PILImage:
        return image.convert("RGB")


@DATA_TRANSFORMS.register("standard_image_transform")
class StandardImageTransform(DataTransform[torch.Tensor]):
    to_rgb: bool = True  # Convert image to RGB if it's in a different mode
    do_normalize: bool = True  # Normalize the image to ImageNet mean and std
    do_resize: bool = True  # Resize the image to 224x224
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "imagenet"
    resize_height: int = 224
    resize_width: int = 224
    image_mean: list[float] | None = None
    image_std: list[float] | None = None

    @property
    def data_model(self) -> type[torch.Tensor]:
        return torch.Tensor

    def model_post_init(self, context) -> None:
        self._transform = None

    def _get_stats(self) -> tuple[list[float], list[float]]:
        if self.stats == "imagenet":
            return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        elif self.stats == "standard":
            return IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
        elif self.stats == "clip":
            return OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
        elif self.stats == "custom":
            if self.image_mean is None or self.image_std is None:
                raise ValueError(
                    "For 'custom' stats, both image_mean and image_std must be provided."
                )
            return self.image_mean, self.image_std
        else:
            raise ValueError(
                f"Unknown stats_type: {self.stats}. Must be one of "
                "'imagenet', 'standard', 'clip', 'custom'."
            )

    def _prepare_image_transform(self):
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        transform = [ToRGB(), ToTensor()]
        if self.do_resize:
            transform += [
                Resize(
                    (self.resize_height, self.resize_width),
                    interpolation=2,  # type: ignore[attr-defined]
                    antialias=True,  # type: ignore[attr-defined]
                )
            ]
        if self.do_normalize:
            mean, std = self._get_stats()
            transform += [Normalize(mean=mean, std=std)]
        transform = Compose(transform)
        return transform

    def __call__(self, image: PILImage) -> torch.Tensor:
        if not self._transform:
            self._transform = self._prepare_image_transform()
        return self._transform(image)  # type: ignore
