"""Common Dataset Types and Enums"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import yaml
from atria_logger import get_logger
from atria_registry._module_base import ModuleConfig
from atria_types import BaseDataInstance
from atria_types._common import DatasetSplitType
from pydantic import ConfigDict

from atria_datasets.core.constants import (
    _DEFAULT_ATRIA_DATASETS_CONFIG_PATH,
    _DEFAULT_ATRIA_DATASETS_METADATA_PATH,
    _DEFAULT_SNAPSHOT_PATH,
)
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_datasets.core.dataset._cached_dataset import CachedDataset
    from atria_datasets.core.dataset._datasets import Dataset
    from atria_datasets.core.storage._storage_managers._deltalake import (
        DeltalakeStorageManager,
    )
    from atria_datasets.core.storage._storage_managers._msgpack import (
        MsgpackStorageManager,
    )

logger = get_logger(__name__)


class DatasetConfig(ModuleConfig):
    model_config = ConfigDict(extra="forbid")
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
        overwrite_existing_cached: bool = False,
        num_processes: int = 8,
        cached_storage_type: FileStorageType = FileStorageType.DELTALAKE,
        enable_cached_splits: bool = True,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
        allowed_keys: set[str] | None = None,
        **kwargs,
    ) -> Dataset | CachedDataset:
        from atria_datasets.core.dataset._dataset_builders import cache, load

        dataset = super().build(**kwargs)
        if enable_cached_splits:
            return cache(
                dataset=dataset,
                data_dir=data_dir,
                split=split,
                access_token=access_token,
                cached_storage_type=cached_storage_type,
                overwrite_existing_cached=overwrite_existing_cached,
                store_artifact_content=store_artifact_content,
                max_cache_image_size=max_cache_image_size,
                num_processes=num_processes,
                allowed_keys=allowed_keys,
            )
        return load(
            dataset=dataset, data_dir=data_dir, split=split, access_token=access_token
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


def _write_yaml(file_path: Path, data: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def _save_dataset_info(
    storage_dir: str, config_name: str, config: dict, metadata: dict
) -> None:
    config_file_path = (
        Path(storage_dir) / config_name / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
    )
    logger.info("Saving dataset configuration to %s", config_file_path)
    _write_yaml(config_file_path, config)

    metadata_file_path = (
        Path(storage_dir) / config_name / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
    )
    logger.info("Saving dataset metadata to %s", metadata_file_path)
    _write_yaml(metadata_file_path, metadata)


def _save_snapshot(
    storage_dir: Path | str,
    config_name: str,
    data_model: type,
    storage_type: FileStorageType,
    dataset_name: str | None,
    dataset_class_name: str,
) -> None:
    snapshot = {
        "storage_type": storage_type.value,
        "data_model": f"{data_model.__module__}.{data_model.__qualname__}",
        "dataset_name": dataset_name,
        "dataset_class_name": dataset_class_name,
        "config_name": config_name,
        "config_hash": config_name.rsplit("-", 1)[-1],
    }
    snapshot_path = Path(storage_dir) / config_name / _DEFAULT_SNAPSHOT_PATH
    logger.info("Saving dataset snapshot to %s", snapshot_path)
    _write_yaml(snapshot_path, snapshot)


def _get_storage_manager(
    cached_storage_type: FileStorageType,
    storage_dir: str,
    config_name: str,
    num_processes: int,
    name_suffix: str = "",
) -> MsgpackStorageManager | DeltalakeStorageManager:
    if cached_storage_type == FileStorageType.DELTALAKE:
        from atria_datasets.core.storage._storage_managers._deltalake import (
            DeltalakeStorageManager,
        )

        return DeltalakeStorageManager(
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
        )
    elif cached_storage_type == FileStorageType.MSGPACK:
        from atria_datasets.core.storage._storage_managers._msgpack import (
            MsgpackStorageManager,
        )

        return MsgpackStorageManager(
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
        )
    else:
        raise ValueError(f"Unsupported storage type: {cached_storage_type}")


T_DatasetConfig = TypeVar("T_DatasetConfig", bound=DatasetConfig)
T_HuggingfaceDatasetConfig = TypeVar(
    "T_HuggingfaceDatasetConfig", bound=HuggingfaceDatasetConfig
)
T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)
