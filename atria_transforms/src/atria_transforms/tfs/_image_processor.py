from __future__ import annotations

from atria_logger import get_logger
from PIL.Image import Image as PILImage

from atria_transforms.core import DataTransform
from atria_transforms.registry import DATA_TRANSFORM

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


@DATA_TRANSFORM.register("image_processor")
class ImageProcessor(DataTransform[PILImage]):
    to_rgb: bool = True  # Convert image to RGB if it's in a different mode
    do_normalize: bool = True  # Normalize the image to ImageNet mean and std
    do_resize: bool = True  # Resize the image to 224x224
    use_imagenet_mean_std: bool = False
    resize_height: int = 224
    resize_width: int = 224
    image_mean: list[float] | None = None
    image_std: list[float] | None = None

    def model_post_init(self, context) -> None:
        self.image_mean = self.image_mean or IMAGENET_STANDARD_MEAN
        self.image_std = self.image_std or IMAGENET_STANDARD_STD
        if self.use_imagenet_mean_std:
            self.image_mean = IMAGENET_DEFAULT_MEAN
            self.image_std = IMAGENET_DEFAULT_STD

        # prepare image transform
        self._transform = None

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
            transform += [Normalize(mean=self.image_mean, std=self.image_std)]
        transform = Compose(transform)
        return transform

    def __call__(self, image: PILImage) -> PILImage:
        if not self._transform:
            self._transform = self._prepare_image_transform()
        return self._transform(image)
