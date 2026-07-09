"""Defines the base Dataset class for Atria datasets."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Generic

from atria_logger import get_logger
from atria_registry import ConfigurableModule
from atria_transforms.core import DataTransform
from atria_types import (
    BaseDataInstance,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
)
from privatekie.datasets.partitioning import PartitionedDataset

from atria_datasets.core.dataset._common import T_BaseDataInstance, T_DatasetConfig
from atria_datasets.core.dataset._dataset_builders import (
    ComposedTransform,
    LoadOutputTransformer,
    _default_data_dir,
    _validate_data_dir,
)
from atria_datasets.core.dataset._exceptions import SplitNotFoundError
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_datasets.core.dataset._cached_dataset import CachedDataset

logger = get_logger(__name__)


class DatasetInputTransform(Generic[T_BaseDataInstance, T_DatasetConfig]):
    def __init__(self, data_model: T_DatasetConfig, config: T_BaseDataInstance):
        self.data_model = data_model
        self.config = config

    def __call__(self, *args, **kwargs):
        assert len(args) == 1, "Expected a single positional argument 'sample'."
        assert len(kwargs) == 0, "No keyword arguments expected."
        sample = args[0]
        if isinstance(sample, self.data_model):
            return sample
        elif isinstance(sample, dict):
            return self.data_model(**sample)
        else:
            raise TypeError(
                f"Cannot convert sample of type {type(sample)} to data model {self.data_model}"
            )


class Dataset(
    ConfigurableModule[T_DatasetConfig], Generic[T_DatasetConfig, T_BaseDataInstance]
):
    """
    Generic base class for datasets in the Atria application.

    This class provides a comprehensive framework for managing datasets with support for:
    - Multiple data splits (train/validation/test)
    - Flexible storage backends (DeltaLake, sharded files)
    - Download management for remote datasets
    - Runtime and preprocessing transformations
    - Dataset versioning and configuration management
    - Hub integration for dataset sharing

    Type Parameters:
        T_BaseDataInstance: The type of data instances this dataset contains
            (must inherit from BaseDataInstance)

    Attributes:
        __data_model__: The data model class used for type validation
        __default_config_path__: Default path for dataset configuration files
        __default_metadata_path__: Default path for dataset metadata files
        __repr_fields__: Fields included in string representation

    Example:
        ```python
        # Create a custom dataset
        class MyDataset(AtriaDataset[DocumentInstance]):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [SplitConfig(split=DatasetSplitType.train, gen_kwargs={})]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Return iterator for the split
                pass


        # Load and use dataset
        dataset = MyDataset(dataset_name="my_dataset")
        dataset.build_split(DatasetSplitType.train)
        for sample in dataset.train:
            print(sample)
        ```
    """

    __abstract__ = True
    __requires_access_token__ = False
    __extract_downloads__ = True
    __data_model__: type[T_BaseDataInstance]
    __input_transform__: type[DatasetInputTransform] = DatasetInputTransform
    __repr_fields__ = {"data_model", "data_dir", "split_iterators"}
    __config__: type[T_DatasetConfig]

    def __init__(self, config: T_DatasetConfig | dict | None = None) -> None:
        super().__init__(config=config)
        self._split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        self._data_dir: str | None = None
        self._downloaded_files: dict[str, Path] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "__abstract__" in cls.__dict__ and cls.__dict__["__abstract__"]:
            return

        data_model = cls.__data_model__
        if data_model is None:
            raise TypeError(
                f"Class '{cls.__name__}' must define a __data_model__ attribute "
                "to specify the type of data instances."
            )
        if not issubclass(data_model, BaseDataInstance):
            raise TypeError(
                f"Class '{cls.__name__}.__data_model__' must be a type, "
                f"got {type(data_model).__name__}: {data_model}"
            )
        assert isinstance(cls.__requires_access_token__, bool), (
            f"Class '{cls.__name__}' must define __requires_access_token__ as a boolean."
        )
        assert isinstance(cls.__extract_downloads__, bool), (
            f"Class '{cls.__name__}' must define __extract_downloads__ as a boolean."
        )

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata containing description, version, and other information."""
        return self._metadata()

    @property
    def downloaded_files(self) -> dict[str, Path]:
        """Dictionary of downloaded file paths, keyed by download key."""
        return self._downloaded_files

    def load(
        self,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        enable_cached_splits: bool = False,
        overwrite_existing_cached: bool = False,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
        num_processes: int = 8,
        cached_storage_type: FileStorageType = FileStorageType.DELTALAKE,
        allowed_keys: set[str] | None = None,
        split_iterator_type: type[SplitIterator] = SplitIterator,
        preprocess_train_transform: DataTransform | None = None,
        preprocess_eval_transform: DataTransform | None = None,
        train_transform: DataTransform | None = None,
        eval_transform: DataTransform | None = None,
        partition_id: int | None = None,
        partition_cache_dir: int | None = None,
    ) -> Dataset | CachedDataset:
        """Entry point for loading a dataset, live or cached.

        enable_cached_splits=False (default): prepares live split iterators via
            _load_splits() (LoadOutputTransformer + train/eval_transform) and
            returns self.
        enable_cached_splits=True: delegates entirely to cache(), which is
            self-sufficient — it checks cache uniqueness via the storage
            manager and builds from scratch if needed — and returns a
            CachedDataset.
        """
        if enable_cached_splits:
            return self.cache(
                data_dir=data_dir,
                split=split,
                access_token=access_token,
                cached_storage_type=cached_storage_type,
                overwrite_existing_cached=overwrite_existing_cached,
                store_artifact_content=store_artifact_content,
                max_cache_image_size=max_cache_image_size,
                num_processes=num_processes,
                allowed_keys=allowed_keys,
                split_iterator_type=split_iterator_type,
                preprocess_train_transform=preprocess_train_transform,
                preprocess_eval_transform=preprocess_eval_transform,
                train_transform=train_transform,
                eval_transform=eval_transform,
                partition_id=partition_id,
                partition_cache_dir=partition_cache_dir,
            )
        return self._load_splits(
            data_dir=data_dir,
            split=split,
            access_token=access_token,
            split_iterator_type=split_iterator_type,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    def _load_splits(
        self,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        split_iterator_type: type[SplitIterator] = SplitIterator,
        train_transform: DataTransform | None = None,
        eval_transform: DataTransform | None = None,
    ) -> Dataset:
        """Prepare live split iterators for direct, uncached iteration.

        Applies LoadOutputTransformer (sample.load()) composed with the given
        train/eval_transform at iteration time. Does not read from or write to
        any on-disk cache — use cache() to persist a snapshot to storage.
        """
        from atria_datasets.core.dataset._dataset_builders import (
            _prepare_downloads,
            _prepare_split,
        )

        resolved = _validate_data_dir(data_dir or _default_data_dir(self))
        self._data_dir = resolved
        self._downloaded_files = _prepare_downloads(self, resolved, access_token)  # type: ignore[assignment]
        split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        for s in self._available_splits():
            if split is not None and s != split:
                continue
            tf = (
                train_transform
                if s == DatasetSplitType.train
                else (eval_transform or train_transform)
            )
            split_iterators[s] = _prepare_split(
                self,
                s,
                resolved,
                split_iterator_type,
                user_transform=tf,
                for_cache=False,
            )
        self._split_iterators = split_iterators
        return self

    def cache(
        self,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        cached_storage_type: FileStorageType = FileStorageType.DELTALAKE,
        overwrite_existing_cached: bool = False,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
        num_processes: int = 8,
        allowed_keys: set[str] | None = None,
        split_iterator_type: type[SplitIterator] = SplitIterator,
        preprocess_train_transform: DataTransform | None = None,
        preprocess_eval_transform: DataTransform | None = None,
        train_transform: DataTransform | None = None,
        eval_transform: DataTransform | None = None,
        partition_id: int | None = None,
        partition_cache_dir: str | None = None,
    ) -> CachedDataset:
        """Build (or reuse) an on-disk cached snapshot of this dataset.

        Fully self-sufficient — does not require load() to have been called
        first. Cache uniqueness is checked exclusively via the storage
        manager's compute_cache_path (which folds its storage_prefix into the
        top-level path, so different storage backends never collide):
        - If a valid cache already exists and overwrite_existing_cached=False,
          it is returned immediately.
        - Otherwise split iterators are (re)built: from scratch via
          dataset._split_iterator(...) whenever a preprocess transform is
          given, or when load() has not populated self._split_iterators yet;
          if load() was already called and no preprocess transform is given,
          the previously-built base iterators are reused (only the
          write-time output transform is rebuilt).

        preprocess_train/eval_transform: baked into the cache at write time
            (PreprocessOutputTransformer path); only supported for MSGPACK.
        train/eval_transform: forwarded to the returned CachedDataset for
            runtime application (never baked into the cache).
        """
        from atria_datasets.core.dataset._cached_dataset import CachedDataset
        from atria_datasets.core.dataset._dataset_builders import (
            _prepare_downloads,
            _prepare_split,
            _resolve_output_data_model,
        )
        from atria_datasets.core.storage._storage_managers._base import StorageManager

        data_dir = _validate_data_dir(data_dir or _default_data_dir(self))
        storage_manager = StorageManager.create(
            cached_storage_type,
            data_dir=data_dir,
            num_processes=num_processes,
            dataset=self,
            preprocess_train_transform=preprocess_train_transform,
            preprocess_eval_transform=preprocess_eval_transform,
        )
        unique_path = storage_manager.storage_dir / storage_manager.config_name

        if (
            unique_path.exists()
            and not overwrite_existing_cached
            and CachedDataset.validate_cache(unique_path)
        ):
            logger.info(f"Loading existing cached dataset from {unique_path}")
            if partition_id is not None:
                return PartitionedDataset(
                    path=unique_path,
                    partition_id=partition_id,
                    partition_cache_dir=partition_cache_dir,
                    allowed_keys=allowed_keys,
                    train_transform=train_transform,
                    eval_transform=eval_transform,
                ).load()
            return CachedDataset(
                path=unique_path,
                allowed_keys=allowed_keys,
                train_transform=train_transform,
                eval_transform=eval_transform,
            ).load()

        if (
            preprocess_train_transform is not None
            or preprocess_eval_transform is not None
        ):
            assert cached_storage_type == FileStorageType.MSGPACK, (
                "Caching with preprocess transforms is only supported for MSGPACK storage type."
            )
            preprocess_tf = preprocess_train_transform or preprocess_eval_transform
            try:
                assert type(preprocess_train_transform) == type(
                    preprocess_eval_transform
                ), (
                    "preprocess_train_transform and preprocess_eval_transform must be of the same type."
                )
                assert issubclass(preprocess_tf.data_model, BaseDataInstance), (
                    "preprocess_transform must implement data_model property returning a BaseDataInstance."
                )
            except NotImplementedError:
                raise ValueError(
                    "preprocess_transform must implement data_model property returning a BaseDataInstance."
                )
        self._data_dir = data_dir
        self._downloaded_files = _prepare_downloads(self, data_dir, access_token)  # type: ignore[assignment]

        reuse_base_iterators = (
            preprocess_train_transform is None and preprocess_eval_transform is None
        )
        split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        for s in self._available_splits():
            if split is not None and s != split:
                continue
            tf = (
                preprocess_train_transform
                if s == DatasetSplitType.train
                else (preprocess_eval_transform or preprocess_train_transform)
            )
            base_iterator = (
                self._split_iterators[s].base_iterator
                if reuse_base_iterators and s in self._split_iterators
                else None
            )
            split_iterators[s] = _prepare_split(
                self,
                s,
                data_dir,
                split_iterator_type,
                store_artifact_content=store_artifact_content,
                resize_images=max_cache_image_size is not None,
                image_max_size=max_cache_image_size,
                user_transform=tf,
                for_cache=True,
                base_iterator=base_iterator,
            )

        for s, split_iterator in split_iterators.items():
            split_exists = storage_manager.split_exists(s)
            if split_exists and overwrite_existing_cached:
                logger.warning(f"Overwriting existing cached split {s.value}")
                storage_manager.purge_split(s)
                split_exists = False
            if not split_exists:
                logger.info(
                    f"Caching split [{s.value}] to {storage_manager.storage_dir}"
                )
                storage_manager.write_split(split_iterator=split_iterator)
            else:
                logger.info(
                    f"Skipping cached split {s.value} at {storage_manager.split_dir(s)}"
                )

        CachedDataset.save_dataset_info(
            str(storage_manager.storage_dir),
            storage_manager.config_name,
            self.config.model_dump(),
            self.metadata.model_dump(),
        )
        output_data_model = (
            _resolve_output_data_model(preprocess_train_transform, self.data_model)
            if preprocess_train_transform is not None
            else self.data_model
        )
        CachedDataset.save_snapshot(
            storage_dir=storage_manager.storage_dir,
            config_name=storage_manager.config_name,
            config_hash=self.config.hash,
            data_model=output_data_model,
            storage_type=cached_storage_type,
            dataset_name=self.config.dataset_name,
            dataset_class_name=self.__class__.__name__,
            train_transform=(
                preprocess_train_transform.to_dict()
                if preprocess_train_transform is not None
                else None
            ),
            eval_transform=(
                preprocess_eval_transform.to_dict()
                if preprocess_eval_transform is not None
                else None
            ),
        )
        return CachedDataset(
            path=unique_path,
            allowed_keys=allowed_keys,
            train_transform=train_transform,
            eval_transform=eval_transform,
        ).load()

    def apply_transforms(
        self,
        train_transform: DataTransform | None = None,
        eval_transform: DataTransform | None = None,
    ) -> None:
        for key, split_iterator in self._split_iterators.items():
            if key == DatasetSplitType.train and train_transform is not None:
                split_iterator.output_transform = (
                    ComposedTransform([LoadOutputTransformer(), train_transform])
                    if train_transform is not None
                    else LoadOutputTransformer()
                )
            elif (
                key in {DatasetSplitType.validation, DatasetSplitType.test}
                and eval_transform is not None
            ):
                split_iterator.output_transform = (
                    ComposedTransform([LoadOutputTransformer(), eval_transform])
                    if eval_transform is not None
                    else LoadOutputTransformer()
                )

    def split_exists(self, split: DatasetSplitType) -> bool:
        """Check if a specific dataset split exists."""
        return split in self._split_iterators

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        """The data model class used for type validation and instantiation."""
        return self.__data_model__

    @property
    def input_transform(self) -> type[T_BaseDataInstance]:
        """The data model class used for type validation and instantiation."""
        return self.__input_transform__(self.data_model, self.config)

    @property
    def train(self) -> SplitIterator[T_BaseDataInstance]:
        """Training split iterator. Returns None if training split is not available."""
        if DatasetSplitType.train not in self._split_iterators:
            raise SplitNotFoundError("Training split iterator is not available. ")
        return self._split_iterators[DatasetSplitType.train]

    @train.setter
    def train(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the training split iterator."""
        self._split_iterators[DatasetSplitType.train] = value

    @property
    def validation(self) -> SplitIterator[T_BaseDataInstance]:
        """Validation split iterator. Returns None if validation split is not available."""
        if DatasetSplitType.validation not in self._split_iterators:
            raise SplitNotFoundError("Validation split iterator is not available. ")
        return self._split_iterators[DatasetSplitType.validation]

    @validation.setter
    def validation(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the validation split iterator."""
        self._split_iterators[DatasetSplitType.validation] = value

    @property
    def test(self) -> SplitIterator[T_BaseDataInstance]:
        """Test split iterator. Returns None if test split is not available."""
        if DatasetSplitType.test not in self._split_iterators:
            raise SplitNotFoundError("Test split iterator is not available. ")
        return self._split_iterators[DatasetSplitType.test]

    @test.setter
    def test(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the test split iterator."""
        self._split_iterators[DatasetSplitType.test] = value

    @property
    def split_iterators(
        self,
    ) -> dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]]:
        """Get all split iterators as a dictionary."""
        return self._split_iterators

    def _download_urls(self) -> dict[str, tuple[str, str]] | list[str]:
        """
        Get the list of URLs for downloading dataset files.

        This method should be overridden by subclasses to provide specific URLs
        for the dataset being implemented.

        Returns:
            List of URLs as strings
        """
        return []

    def _custom_download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        """This method can be overridden by subclasses to implement custom download logic.

        Args:
            data_dir: Directory to save downloaded files
            access_token: Authentication token for private resources

        Returns:
            Dictionary mapping download keys to downloaded file paths
        """
        raise NotImplementedError(
            "Subclasses must implement the `_custom_download` method to handle "
            "specific download logic."
        )

    @abstractmethod
    def _metadata(self) -> DatasetMetadata:
        """
        Create and return dataset metadata.

        Subclasses should override this method to provide specific metadata
        including description, version, license, and other relevant information.

        Returns:
            DatasetMetadata object with dataset information
        """
        raise NotImplementedError("Subclasses must implement the `_metadata` method.")

    @abstractmethod
    def _available_splits(self) -> list[DatasetSplitType]:
        """
        List available dataset splits.

        Subclasses should override this method to return the splits that are
        available for the dataset (e.g., train, validation, test).

        Returns:
            List of DatasetSplitType values representing available splits
        """
        raise NotImplementedError(
            "Subclasses must implement the `_available_splits` method."
        )

    @abstractmethod
    def _split_iterator(self, split: DatasetSplitType, data_dir: str) -> Iterable:
        """
        Create an iterator for a specific dataset split.

        Args:
            split: The dataset split to create iterator for
            **kwargs: Additional arguments from split configuration

        Returns:
            Iterator or generator yielding data samples for the split

        Note:
            Subclasses must implement this method to define how to iterate
            over the data for each split. The iterator should yield raw data
            that will be transformed by _input_transform.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_split_iterator` method to provide "
            "an iterator for the specified dataset split."
        )


class ImageDataset(Dataset[T_DatasetConfig, ImageInstance], Generic[T_DatasetConfig]):
    """
    Specialized dataset class for handling image datasets.

    This class inherits from AtriaDataset and is specifically typed for ImageInstance
    data models, providing type safety and specialized functionality for image data.

    The class automatically handles:
    - Image-specific data validation
    - Proper type hints for image data
    - Integration with image processing pipelines

    Example:
        ```python
        class CustomImageDataset(AtriaImageDataset):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [
                    SplitConfig(
                        split=DatasetSplitType.train,
                        gen_kwargs={"image_dir": f"{data_dir}/train"},
                    )
                ]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Yield image data samples
                pass
        ```
    """

    __abstract__: bool = True
    __data_model__ = ImageInstance


class DocumentDataset(
    Dataset[T_DatasetConfig, DocumentInstance], Generic[T_DatasetConfig]
):
    """
    Specialized dataset class for handling document datasets.

    This class inherits from AtriaDataset and is specifically typed for DocumentInstance
    data models, providing type safety and specialized functionality for document data.

    The class automatically handles:
    - Document-specific data validation
    - Proper type hints for document data
    - Integration with text processing pipelines

    Example:
        ```python
        class CustomDocumentDataset(AtriaDocumentDataset):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [
                    SplitConfig(
                        split=DatasetSplitType.train,
                        gen_kwargs={"text_dir": f"{data_dir}/train"},
                    )
                ]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Yield document data samples
                pass
        ```
    """

    __abstract__: bool = True
    __data_model__ = DocumentInstance
