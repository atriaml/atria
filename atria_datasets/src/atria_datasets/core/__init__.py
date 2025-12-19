# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from atria_datasets.core.dataset._common import DatasetConfig
    from atria_datasets.core.dataset._datasets import Dataset
    from atria_datasets.core.dataset_splitters.standard_splitter import StandardSplitter

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "dataset._common": ["DatasetConfig"],
        "dataset._datasets": ["Dataset"],
        "dataset_splitters.standard_splitter": ["StandardSplitter"],
    },
)
