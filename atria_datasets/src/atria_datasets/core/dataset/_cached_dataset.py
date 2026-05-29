"""CachedDataset - immutable, file-backed dataset loaded from a snapshot directory."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Generic

import yaml
from atria_logger import get_logger
from atria_types import DatasetMetadata, DatasetSplitType, RepresentationMixin

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
    from atria_logger import get_logger
    from atria_transforms.core import DataTransform

logger = get_logger(__name__)


class CachedDataset(RepresentationMixin, Generic[T_BaseDataInstance]):
    """Immutable, file-backed dataset produced by cache().

    All state is loaded lazily from the snapshot directory on disk.
    Upload and download hub operations are available directly on this class.
    """

    __repr_fields__ = {"data_model", "data_dir", "split_iterators", "metadata"}

    def __init__(self, path: Path | str, allowed_keys: set[str] | None = None) -> None:
        self._path = Path(path)
        self.__snapshot: dict | None = None
        self.__metadata: DatasetMetadata | None = None
        self.__data_model: type[T_BaseDataInstance] | None = None
        self.__split_iterators: dict[DatasetSplitType, SplitIterator] | None = None
        self._allowed_keys: set[str] | None = allowed_keys

    @classmethod
    def load_from_hub(
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

        return DatasetHubOps.load_from_hub(
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
    ) -> dict[str, str]:
        """Upload this frozen cached dataset snapshot to Atria Hub."""
        from atria_datasets.core.dataset._ops import DatasetHubOps

        return DatasetHubOps(self).upload_to_hub(
            name=name,
            branch=branch,
            is_public=is_public,
            overwrite_existing=overwrite_existing,
        )
    
    @classmethod
    def validate_cache(cls, path: Path | str) -> bool:
        """Validate that the given path contains a valid cached dataset snapshot."""
        path = Path(path)
        snapshot_file = path / _DEFAULT_SNAPSHOT_PATH
        if not snapshot_file.exists():
            return False
        with open(snapshot_file) as f:
            snapshot = yaml.safe_load(f)
        required_keys = {"dataset_class_name", "data_model", "storage_type", "config_name", "config_hash"}
        missing_keys = required_keys - snapshot.keys()
        if missing_keys:
            return False
        return True

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

    def process(
        self,
        train_transform: DataTransform,
        eval_transform: DataTransform | None = None,
        split: DatasetSplitType | None = None,
        cached_storage_type: FileStorageType = FileStorageType.DELTALAKE,
        overwrite_existing_cached: bool = False,
        num_processes: int = 8,
        allowed_keys: set[str] | None = None,
    ) -> CachedDataset:
        """Apply transforms to each split and return a new file-backed CachedDataset."""
        from atria_datasets.core.dataset._common import (
            _get_storage_manager,
            _save_dataset_info,
            _save_snapshot,
        )
        from atria_datasets.core.dataset._dataset_builders import (
            _get_combined_transform_hash,
            _resolve_output_data_model,
        )

        storage_dir = self._path.parent
        unique_config_name = self._path.name + "-" + _get_combined_transform_hash(train_transform, eval_transform)

        if train_transform is not None or eval_transform is not None:
            assert cached_storage_type == FileStorageType.MSGPACK, (
                "Process with transforms is currently only supported for Msgpack storage. Please set cached_storage_type=FileStorageType.MSGPACK."
            )
            # try:
            #     assert type(train_transform) == type(eval_transform), "train_transform and eval_transform must be of the same type."
            #     data_model = train_transform.data_model
            #     if not issubclass(data_model, BaseDataInstance):
            #         raise ValueError(
            #             f"Output transform data_model must be a subclass of BaseDataInstance. Got: {data_model}"
            #         )
            # except NotImplementedError:
            #     raise ValueError("transform must implement data_model property that must return a BaseDataInstance.")

        storage_manager = _get_storage_manager(
            cached_storage_type, str(storage_dir), unique_config_name, num_processes
        )

        splits_to_cache: list[DatasetSplitType] = []
        for s in list(self.split_iterators):
            if split is not None and s != split:
                continue
            split_exists = storage_manager.split_exists(split=s)
            if split_exists and overwrite_existing_cached:
                logger.warning(f"Overwriting existing cached split {s.value}")
                storage_manager.purge_split(s)
                split_exists = False
            if not split_exists:
                splits_to_cache.append(s)
            else:
                logger.info(f"Loading cached split {s.value} from {storage_manager.split_dir(s)}")

        for s in splits_to_cache:
            logger.info(f"Caching split [{s.value}] to {storage_dir}")
            tf = train_transform if s == DatasetSplitType.train else (eval_transform or train_transform)
            split_iterator = self.split_iterators[s]
            split_iterator.output_transform = tf
            storage_manager.write_split(split_iterator=split_iterator)

        _save_dataset_info(
            str(storage_dir), unique_config_name,
            self.config,
            self.metadata.model_dump() if self.metadata is not None else {},
        )
        output_data_model = _resolve_output_data_model(train_transform, self.data_model)
        _save_snapshot(
            storage_dir=storage_dir,
            config_name=unique_config_name,
            data_model=output_data_model,
            storage_type=cached_storage_type,
            dataset_name=self.dataset_name,
            dataset_class_name=self.dataset_class_name,
            train_transform=train_transform.to_dict(),
            eval_transform=eval_transform.to_dict() if eval_transform is not None else None,
        )
        return CachedDataset(path=storage_dir / unique_config_name, allowed_keys=allowed_keys)

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
            split: storage_manager.read_split(
                split=split, data_model=self.data_model, allowed_keys=self._allowed_keys
            )
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
