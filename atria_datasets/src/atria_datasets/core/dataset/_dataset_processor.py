"""Dataset Builder Module"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from atria_logger import get_logger
from atria_transforms.core import DataTransform
from atria_types import DatasetSplitType

from atria_datasets.core.constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR,
)
from atria_datasets.core.dataset._common import DatasetConfig, _get_storage_manager
from atria_datasets.core.dataset._dataset_builders import DefaultOutputTransformer
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

logger = get_logger(__name__)


class ComposedTransform:
    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, sample):
        for transform in self._transforms:
            sample = transform(sample)
        return sample


class DatasetProcessor:
    def __init__(
        self,
        dataset: Dataset,
        train_transform: Callable | DataTransform,
        eval_transform: Callable | DataTransform | None = None,
        split: DatasetSplitType | None = None,
        data_dir: str | None = None,
        cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
        overwrite_existing_cached: bool = False,
        store_artifact_content: bool = True,
        max_cache_image_size: int | tuple[int, int] | None = None,
        num_processes: int = 8,
    ):
        self._dataset = dataset
        self._split = split
        self._train_transform = train_transform
        self._eval_transform = eval_transform
        self._data_dir = self._validate_data_dir(
            data_dir or (_DEFAULT_ATRIA_DATASETS_CACHE_DIR / self.dataset_name)
        )
        self._cached_storage_type = cached_storage_type
        self._overwrite_existing_cached = overwrite_existing_cached
        self._store_artifact_content = store_artifact_content
        self._max_cache_image_size = max_cache_image_size
        self._num_processes = num_processes
        self._storage_dir = (
            Path(self._data_dir) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
        )

    @property
    def dataset(self) -> Dataset:
        """Get the dataset being built."""
        return self._dataset

    @property
    def dataset_name(self) -> str:
        """Get the name of the dataset being built."""
        return (
            self.dataset_config.dataset_name
            if self.dataset_config.dataset_name
            else self._dataset.__class__.__name__
        )

    @property
    def dataset_config(self) -> DatasetConfig:
        """Get the configuration of the dataset being built."""
        return self._dataset.config

    @property
    def dataset_metadata(self):
        """Get the metadata of the dataset being built."""
        return self._dataset.metadata

    def _validate_data_dir(self, data_dir: str | Path) -> str:
        data_dir = Path(data_dir)

        if data_dir.exists():
            assert data_dir.is_dir(), (
                f"Data directory `{data_dir.absolute()}` exists but is not a directory."
            )
        else:
            logger.warning(
                f"Data directory `{data_dir.absolute()}` does not exist. Creating it."
            )
            data_dir.mkdir(parents=True, exist_ok=True)

        return str(data_dir)

    def _prepare_split(
        self,
        split: DatasetSplitType,
        transform: Callable | DataTransform,
        store_artifact_content: bool = True,
        resize_images: bool = False,
        image_max_size: int | tuple[int, int] | None = None,
    ) -> SplitIterator:
        if split not in self.dataset._available_splits():
            raise ValueError(f"Split {split} is not available in the dataset.")
        split_iterator = self.dataset._split_iterators[split]

        # get detault output transform for the split
        output_transform = DefaultOutputTransformer(
            data_dir=self._data_dir,
            store_artifact_content=store_artifact_content,
            resize_images=resize_images,
            image_max_size=image_max_size,
        )

        # compose the transforms now
        split_iterator.output_transform = ComposedTransform(
            transforms=[output_transform, transform]
        )
        return split_iterator

    def _get_transform_hash(self, transform: Callable | DataTransform) -> str:
        """Get the hash of the transform applied to the dataset being built."""
        if isinstance(transform, DataTransform):
            return transform.hash
        elif callable(transform):
            return "default"
        else:
            raise TypeError(
                f"Transform of type {type(transform)} is not supported for hashing."
            )

    def process_splits(self) -> dict[DatasetSplitType, SplitIterator]:
        """Prepare cached splits using DeltaLake / Msgpack storage."""

        split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        for split in self.dataset._available_splits():
            if self._split is not None and split != self._split:
                continue
            split_iterators[split] = self.process_split(split)

        return split_iterators

    def process_split(self, split: DatasetSplitType) -> SplitIterator:
        """Prepare cached splits using DeltaLake / Msgpack storage."""
        # Get the appropriate transform
        transform = (
            self._train_transform
            if split == DatasetSplitType.train
            else self._eval_transform
        )
        data_model = None
        if isinstance(transform, DataTransform):
            data_model = transform.data_model
        assert transform is not None, (
            f"Transform for split {split.value} is not provided."
        )

        # Get storage manager
        storage_manager = _get_storage_manager(
            self._cached_storage_type,
            storage_dir=str(self._storage_dir),
            config_name=self.dataset_config.config_name
            + "-"
            + self.dataset_config.hash,
            num_processes=self._num_processes,
            name_suffix=self._get_transform_hash(transform),
        )

        # Check if split exists
        split_exists = storage_manager.split_exists(split=split)
        if split_exists:
            if self._overwrite_existing_cached:
                logger.warning(f"Overwriting existing cached split {split.value}")
                storage_manager.purge_split(split)
                split_exists = False
            else:
                logger.info(
                    f"Loading cached split {split.value} from {storage_manager.split_dir(split)}"
                )

                return storage_manager.read_split(split=split, data_model=data_model)

        split_iterator = self._prepare_split(
            split=split,
            transform=transform,
            store_artifact_content=self._store_artifact_content,
            resize_images=self._max_cache_image_size is not None,
            image_max_size=self._max_cache_image_size,
        )
        logger.info(
            f"Caching split [{split.value}] to {self._storage_dir} with max_len={split_iterator._max_len}"
        )
        storage_manager.write_split(split_iterator=split_iterator)
        return storage_manager.read_split(split=split, data_model=data_model)
