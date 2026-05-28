"""CachedDataset - immutable, file-backed dataset loaded from a snapshot directory."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Generic

import yaml
from atria_logger import get_logger
from atria_types import DatasetMetadata, DatasetSplitType

from atria_datasets.core.constants import (
    _DEFAULT_ATRIA_DATASETS_CONFIG_PATH,
    _DEFAULT_ATRIA_DATASETS_METADATA_PATH,
    _DEFAULT_SNAPSHOT_PATH,
)
from atria_datasets.core.dataset._common import T_BaseDataInstance
from atria_datasets.core.dataset._exceptions import SplitNotFoundError
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_hub.utilities import get_logger
    from atriax_client.models.dataset import Dataset as DatasetInfo

    from atria_datasets.core.dataset._cached_dataset import CachedDataset

logger = get_logger(__name__)


class CachedDataset(Generic[T_BaseDataInstance]):
    """Immutable, file-backed dataset produced by cache().

    All state is loaded lazily from the snapshot directory on disk.
    Upload and download hub operations are available directly on this class.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self.__snapshot: dict | None = None
        self.__metadata: DatasetMetadata | None = None
        self.__data_model: type[T_BaseDataInstance] | None = None
        self.__split_iterators: dict[DatasetSplitType, SplitIterator] | None = None

    @classmethod
    def download_from_hub(
        cls,
        name: str,
        username: str | None = None,
        branch: str = "main",
        config_dir: str | None = None,
        storage_dir: str | Path | None = None,
        overwrite_existing: bool = False,
    ) -> CachedDataset:
        """Download a frozen cached dataset snapshot from Atria Hub.

        This is a convenience wrapper around `DatasetHubOps.download_from_hub`.
        """
        from atria_datasets.core.dataset._ops import DatasetHubOps

        return DatasetHubOps.download_from_hub(
            name=name,
            username=username,
            branch=branch,
            config_dir=config_dir,
            storage_dir=storage_dir,
            overwrite_existing=overwrite_existing,
        )

    def upload_to_hub(
        self,
        name: str | None = None,
        branch: str = "main",
        is_public: bool = False,
        overwrite_existing: bool = False,
    ) -> DatasetInfo:
        """Upload this frozen cached dataset snapshot to Atria Hub."""
        from atria_datasets.core.dataset._ops import DatasetHubOps

        return DatasetHubOps(self).upload_to_hub(
            name=name,
            branch=branch,
            is_public=is_public,
            overwrite_existing=overwrite_existing,
        )

    @property
    def _snapshot(self) -> dict:
        if self.__snapshot is None:
            snapshot_file = self._path / _DEFAULT_SNAPSHOT_PATH
            with open(snapshot_file) as f:
                self.__snapshot = yaml.safe_load(f)
        assert self.__snapshot is not None
        return self.__snapshot

    @property
    def dataset_name(self) -> str | None:
        return self._snapshot.get("dataset_name")

    @property
    def dataset_class_name(self) -> str:
        return self._snapshot["dataset_class_name"]

    @property
    def storage_type(self) -> FileStorageType:
        return FileStorageType(self._snapshot["storage_type"])

    @property
    def config_name(self) -> str:
        return self._snapshot["config_name"]

    @property
    def config_hash(self) -> str:
        return self._snapshot["config_hash"]

    @property
    def config(self) -> dict:
        config_path = self._path / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
        with open(config_path) as f:
            return yaml.safe_load(f)

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        if self.__data_model is None:
            fqn = self._snapshot["data_model"]
            module_name, class_name = fqn.rsplit(".", 1)
            module = importlib.import_module(module_name)
            self.__data_model = getattr(module, class_name)
        assert self.__data_model is not None
        return self.__data_model

    @property
    def metadata(self) -> DatasetMetadata | None:
        if self.__metadata is None:
            metadata_path = self._path / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.__metadata = DatasetMetadata(**yaml.safe_load(f))
        return self.__metadata

    def split_exists(self, split: DatasetSplitType) -> bool:
        return split in self.split_iterators

    @property
    def split_iterators(
        self,
    ) -> dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]]:
        if self.__split_iterators is None:
            self.__split_iterators = self._load_split_iterators()
        return self.__split_iterators

    def _load_split_iterators(self) -> dict[DatasetSplitType, SplitIterator]:
        from atria_datasets.core.dataset._common import _get_storage_manager

        storage_manager = _get_storage_manager(
            cached_storage_type=self.storage_type,
            storage_dir=str(self._path.parent),
            config_name=self._path.name,
            num_processes=1,
        )
        return {
            split: storage_manager.read_split(split=split, data_model=self.data_model)
            for split in DatasetSplitType
            if storage_manager.split_exists(split)
        }

    @property
    def train(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.train not in self.split_iterators:
            raise SplitNotFoundError("Training split iterator is not available.")
        return self.split_iterators[DatasetSplitType.train]

    @train.setter
    def train(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self.split_iterators[DatasetSplitType.train] = value

    @property
    def validation(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.validation not in self.split_iterators:
            raise SplitNotFoundError("Validation split iterator is not available.")
        return self.split_iterators[DatasetSplitType.validation]

    @validation.setter
    def validation(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self.split_iterators[DatasetSplitType.validation] = value

    @property
    def test(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.test not in self.split_iterators:
            raise SplitNotFoundError("Test split iterator is not available.")
        return self.split_iterators[DatasetSplitType.test]

    @test.setter
    def test(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self.split_iterators[DatasetSplitType.test] = value
