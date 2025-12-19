# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._data_types._base import TensorDataModel
    from ._tfs._base import DataTransform


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_data_types._base": ["TensorDataModel"],
        "_tfs._base": ["DataTransform"],
    },
)
