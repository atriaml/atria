from collections.abc import Callable
from typing import TYPE_CHECKING

from atria_core.transforms.base import DataTransform
from atria_registry.registry_config import RegistryConfig
from pydantic import ConfigDict

from atria_transforms.registry import DATA_TRANSFORM

if TYPE_CHECKING:
    import torch


@DATA_TRANSFORM.register(
    "image",
    configs=[
        RegistryConfig(name="resize", tf="Resize", interpolation=2, size=(224, 224)),
        RegistryConfig(name="to_tensor", tf="ToTensor"),
        RegistryConfig(name="center_crop", tf="CenterCrop"),
        RegistryConfig(
            name="normalize",
            tf="Normalize",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        RegistryConfig(name="random_horizontal_flip", tf="RandomHorizontalFlip"),
    ],
)
class TorchvisionTransform(DataTransform):  # or inherit from DataTransform if needed
    model_config = ConfigDict(extra="allow")
    tf: str
    _built_tf: Callable

    def _lazy_post_init(self) -> None:
        from atria_core.utilities.imports import _resolve_module_from_path

        self._built_tf = _resolve_module_from_path(f"torchvision.transforms.{self.tf}")(
            **self.model_extra
        )

    def _apply_transforms(self, image: "torch.Tensor") -> "torch.Tensor":
        return self._built_tf(image)
