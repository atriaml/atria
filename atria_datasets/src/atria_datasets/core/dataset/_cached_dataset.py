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
from atria_datasets.core.dataset._common import DatasetConfig, T_BaseDataInstance
from atria_datasets.core.dataset._exceptions import SplitNotFoundError
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_transforms.core import DataTransform

logger = get_logger(__name__)


class CachedDataset(RepresentationMixin, Generic[T_BaseDataInstance]):
    """Immutable, file-backed dataset produced by cache().

    Construction is lightweight — call load() to perform all I/O and populate
    state. Hub upload/download operations are available directly on this class.
    """

    __repr_fields__ = {"data_model", "data_dir", "split_iterators", "metadata"}

    def __init__(
        self,
        path: Path | str,
        allowed_keys: set[str] | None = None,
        train_transform: DataTransform | None = None,
        eval_transform: DataTransform | None = None,
    ) -> None:
        self._path = Path(path)
        self._allowed_keys = allowed_keys
        self._train_transform = train_transform
        self._eval_transform = eval_transform
        # Populated by load()
        self._snapshot_data: dict | None = None
        self._metadata_data: DatasetMetadata | None = None
        self._data_model_cls: type[T_BaseDataInstance] | None = None
        self._split_iterators_data: dict[DatasetSplitType, SplitIterator] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> CachedDataset:
        """Load all state from disk: snapshot, metadata, data model, split iterators."""
        snapshot_file = self._path / _DEFAULT_SNAPSHOT_PATH
        with open(snapshot_file) as f:
            self._snapshot_data = yaml.safe_load(f)
        assert self._snapshot_data is not None, (
            f"Snapshot file is empty: {snapshot_file}"
        )

        fqn = self._snapshot_data["data_model"]
        module_name, class_name = fqn.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self._data_model_cls = getattr(module, class_name)

        metadata_path = self._path / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata_data = DatasetMetadata(**yaml.safe_load(f))

        self._split_iterators_data = self._build_split_iterators()
        return self

    # ------------------------------------------------------------------
    # Hub ops
    # ------------------------------------------------------------------

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
        """Download a frozen cached dataset snapshot from Atria Hub."""
        from atria_datasets.core.dataset._hub_ops import DatasetHubOps

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
        from atria_datasets.core.dataset._hub_ops import DatasetHubOps

        return DatasetHubOps(self).upload_to_hub(
            name=name,
            branch=branch,
            is_public=is_public,
            overwrite_existing=overwrite_existing,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def validate_cache(cls, path: Path | str) -> bool:
        """Validate that the given path contains a valid cached dataset snapshot."""
        path = Path(path)
        snapshot_file = path / _DEFAULT_SNAPSHOT_PATH
        if not snapshot_file.exists():
            return False
        with open(snapshot_file) as f:
            snapshot = yaml.safe_load(f)
        required_keys = {
            "dataset_class_name",
            "data_model",
            "storage_type",
            "config_name",
            "config_hash",
        }
        return not (required_keys - snapshot.keys())

    # ------------------------------------------------------------------
    # Properties (plain getters — require load() to have been called)
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if self._snapshot_data is None:
            raise RuntimeError("CachedDataset state not loaded. Call .load() first.")

    @property
    def data_dir(self) -> Path:
        return self._path

    @property
    def dataset_name(self) -> str | None:
        self._require_loaded()
        return self._snapshot_data.get("dataset_name")  # type: ignore[union-attr]

    @property
    def dataset_class_name(self) -> str:
        self._require_loaded()
        return self._snapshot_data["dataset_class_name"]  # type: ignore[index]

    @property
    def storage_type(self) -> FileStorageType:
        self._require_loaded()
        return FileStorageType(self._snapshot_data["storage_type"])  # type: ignore[index]

    @property
    def config_name(self) -> str:
        self._require_loaded()
        return self._snapshot_data["config_name"]  # type: ignore[index]

    @property
    def config_hash(self) -> str:
        self._require_loaded()
        return self._snapshot_data["config_hash"]  # type: ignore[index]

    @property
    def config(self) -> DatasetConfig:
        config_path = self._path / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
        with open(config_path) as f:
            return DatasetConfig.model_validate(yaml.safe_load(f))

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        self._require_loaded()
        return self._data_model_cls  # type: ignore[return-value]

    @property
    def metadata(self) -> DatasetMetadata | None:
        self._require_loaded()
        return self._metadata_data

    @property
    def split_iterators(
        self,
    ) -> dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]]:
        self._require_loaded()
        return self._split_iterators_data  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_split_iterators(self) -> dict[DatasetSplitType, SplitIterator]:
        from atria_datasets.core.dataset._common import _get_storage_manager
        from atria_datasets.core.dataset._dataset_builders import (
            ComposedTransform,
            LoadOutputTransformer,
        )

        storage_manager = _get_storage_manager(
            data_dir=str(self._path.parent),
            cached_storage_type=self.storage_type,
            storage_dir=str(self._path.parent),
            config_name=self._path.name,
            num_processes=1,
        )
        result: dict[DatasetSplitType, SplitIterator] = {}
        for split in DatasetSplitType:
            if not storage_manager.split_exists(split):
                continue
            iterator = storage_manager.read_split(
                split=split, data_model=self.data_model, allowed_keys=self._allowed_keys
            )
            tf = (
                self._train_transform
                if split == DatasetSplitType.train
                else (self._eval_transform or self._train_transform)
            )
            iterator.output_transform = (
                ComposedTransform([LoadOutputTransformer(), tf])
                if tf is not None
                else LoadOutputTransformer()
            )
            result[split] = iterator
        return result

    # ------------------------------------------------------------------
    # Split access
    # ------------------------------------------------------------------

    def split_exists(self, split: DatasetSplitType) -> bool:
        return split in self.split_iterators

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
