# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._base import ModelBuilder
    from ._common import FrozenLayers, ModelBuilderType
    from ._timm import TimmModelBuilder
    from ._torchvision import TorchvisionModelBuilder
    from ._transformers import TransformersModelBuilder


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_base": ["ModelBuilder"],
        "_common": ["FrozenLayers", "ModelBuilderType"],
        "_torchvision": ["TorchvisionModelBuilder"],
        "_timm": ["TimmModelBuilder"],
        "_transformers": ["TransformersModelBuilder"],
    },
)
