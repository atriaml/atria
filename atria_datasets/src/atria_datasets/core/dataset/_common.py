"""Common Dataset Types and Enums"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, TypeVar

from atria_logger import get_logger
from atria_registry._module_base import ModuleConfig
from atria_types import BaseDataInstance
from atria_types._common import DatasetSplitType
from pydantic import ConfigDict

from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_transforms.core import DataTransform

    from atria_datasets.core.dataset._cached_dataset import CachedDataset
    from atria_datasets.core.dataset._datasets import Dataset

logger = get_logger(__name__)


class DatasetConfig(ModuleConfig):
    model_config = ConfigDict(extra="allow")
    dataset_name: str | None = None
    config_name: str = "default"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    seed: int = 42

    def build(
        self,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        enable_cached_splits: bool = True,
        overwrite_existing_cached: bool = False,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
        num_processes: int = 8,
        cached_storage_type: FileStorageType = FileStorageType.DELTALAKE,
        allowed_keys: set[str] | None = None,
        preprocess_train_transform: DataTransform | None = None,
        preprocess_eval_transform: DataTransform | None = None,
        train_transform: DataTransform | None = None,
        eval_transform: DataTransform | None = None,
        **kwargs,
    ) -> Dataset | CachedDataset:
        dataset = super().build(**kwargs)
        return dataset.load(
            data_dir=data_dir,
            split=split,
            access_token=access_token,
            enable_cached_splits=enable_cached_splits,
            overwrite_existing_cached=overwrite_existing_cached,
            store_artifact_content=store_artifact_content,
            max_cache_image_size=max_cache_image_size,
            num_processes=num_processes,
            cached_storage_type=cached_storage_type,
            allowed_keys=allowed_keys,
            preprocess_train_transform=preprocess_train_transform,
            preprocess_eval_transform=preprocess_eval_transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )


class HuggingfaceDatasetConfig(DatasetConfig):
    hf_repo: str
    hf_config_name: str


class DatasetLoadingMode(str, enum.Enum):
    """
    Enum to represent the streaming mode of the dataset.

    Attributes:
        LOCAL: Dataset is downloaded and stored locally.
        STREAMING: Dataset is streamed directly from the Atria Hub.
    """

    in_memory = "in_memory"
    local_streaming = "local_streaming"
    online_streaming = "online_streaming"


T_DatasetConfig = TypeVar("T_DatasetConfig", bound=DatasetConfig)
T_HuggingfaceDatasetConfig = TypeVar(
    "T_HuggingfaceDatasetConfig", bound=HuggingfaceDatasetConfig
)
T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)
