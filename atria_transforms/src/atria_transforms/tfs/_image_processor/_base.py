from __future__ import annotations

from atria_logger import get_logger
from atria_types import ImageInstance
from atria_types._generic._annotations import AnnotationType
from pydantic import Field

from atria_transforms.core import DataTransform
from atria_transforms.data_types import ImageTensorDataModel
from atria_transforms.registry import DATA_TRANSFORMS
from atria_transforms.tfs import StandardImageTransform

logger = get_logger(__name__)


@DATA_TRANSFORMS.register("image_processor")
class ImageProcessor(DataTransform[ImageTensorDataModel]):
    tf: StandardImageTransform = Field(default_factory=StandardImageTransform)

    def __call__(
        self, image_instance: ImageInstance
    ) -> ImageTensorDataModel | list[ImageTensorDataModel]:
        import torch

        assert image_instance.image.content is not None, "Image content is None."
        image_tensor = self.tf(image_instance.image.content)
        label = image_instance.get_annotation_by_type(
            AnnotationType.classification
        ).label.value
        label = torch.tensor(label, dtype=torch.long)
        return self._output_transform(
            image_instance=image_instance, image=image_tensor, label=label
        )

    def _output_transform(
        self, image_instance: ImageInstance, **kwargs
    ) -> ImageTensorDataModel:
        return ImageTensorDataModel(
            index=image_instance.index, sample_id=image_instance.sample_id, **kwargs
        )
