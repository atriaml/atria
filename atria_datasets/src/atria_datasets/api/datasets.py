"""API functions for loading and preprocessing datasets."""

from __future__ import annotations

from atria_logger import get_logger

from atria_datasets.core import Dataset, DatasetConfig
from atria_datasets.registry import DATASET

logger = get_logger(__name__)


def load_dataset(dataset_name: str, **kwargs) -> Dataset:
    return DATASET.load_module(dataset_name, **kwargs)


def load_dataset_config(dataset_name: str, **kwargs) -> DatasetConfig:
    _, config = DATASET.load_module_config(dataset_name, **kwargs)
    return config


# def load_preprocessed_dataset(
#     dataset_name: str,
#     split: str | None = None,
#     transform_name: str | None = None,
#     **transforms_kwargs,
# ) -> Dataset:
#     # load raw dataset
#     dataset = load_dataset(dataset_name=dataset_name, split=split)

#     # preprocess dataset
#     dataset = preprocess_dataset(
#         dataset=dataset,
#         force_overwrite=False,
#         transform_type=transform_type,
#         **transforms_kwargs,
#     )

#     return dataset
