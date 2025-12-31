from __future__ import annotations

from collections.abc import Callable

from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.core.dataset._common import DatasetConfig
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset_splitters._standard_splitter import StandardSplitter
from atria_datasets.core.storage.utilities import FileStorageType
from atria_logger import get_logger
from atria_registry._module_base import BaseModel
from atria_types import DatasetSplitType
from atria_types._utilities._repr import RepresentationMixin
from pydantic import ConfigDict, Field

logger = get_logger(__name__)


class DataConfig(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    dataset_config: DatasetConfig = Field(
        default_factory=lambda: load_dataset_config("cifar10/1k")
    )
    data_dir: str | None = None

    # caching config
    access_token: str | None = None
    overwrite_existing_cached: bool = False
    allowed_keys: set[str] | None = None
    num_processes: int = 8
    cached_storage_type: FileStorageType = FileStorageType.MSGPACK
    enable_cached_splits: bool = True
    store_artifact_content: bool = True
    max_cache_image_size: int | None = None

    # dataloader config
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    # dataset split args
    splitting_enabled: bool = True
    split_ratio: float = 0.9

    # preprocess transforms
    preprocess_train_transform: Callable | None = None
    preprocess_eval_transform: Callable | None = None
    preprocess_max_cache_image_size: int | None = None

    def build_dataset(self) -> Dataset:
        dataset = self.dataset_config.build(
            data_dir=self.data_dir,
            access_token=self.access_token,
            overwrite_existing_cached=self.overwrite_existing_cached,
            allowed_keys=self.allowed_keys,
            num_processes=self.num_processes,
            cached_storage_type=self.cached_storage_type,
            enable_cached_splits=self.enable_cached_splits,
            store_artifact_content=self.store_artifact_content,
            max_cache_image_size=self.max_cache_image_size,
        )
        if (
            DatasetSplitType.validation not in dataset.split_iterators
            and self.splitting_enabled
        ):
            dataset_splitter = StandardSplitter(
                split_ratio=self.split_ratio, shuffle=True
            )
            train, validation = dataset_splitter(dataset.train)
            dataset.train = train
            dataset.validation = validation

            assert dataset.train is not None, (
                "Training split is None in the loaded dataset"
            )  # for our experiments we always make sure we have validation split present
            assert dataset.validation is not None, (
                "Validation split is None in the loaded dataset"
            )  # for our experiments we always make sure we have validation split present

        return dataset
