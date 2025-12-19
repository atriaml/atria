# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._document import DocumentTensorDataModel
    from ._image import ImageTensorDataModel


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_document": ["DocumentTensorDataModel"],
        "_image": ["ImageTensorDataModel"],
    },
)
