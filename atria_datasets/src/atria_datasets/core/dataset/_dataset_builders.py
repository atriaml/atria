"""Dataset builder functions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from atria_logger import get_logger
from atria_transforms.core import DataTransform
from atria_types import DatasetSplitType
from atria_types._data_instance._base import BaseDataInstance
from atria_types._data_instance._document_instance import DocumentInstance
from atria_types._data_instance._image_instance import ImageInstance

from atria_datasets.core.constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR,
    _DEFAULT_DOWNLOAD_PATH,
)
from atria_datasets.core.dataset._common import (
    _get_storage_manager,
    _save_dataset_info,
    _save_snapshot,
)
from atria_datasets.core.dataset._cached_dataset import CachedDataset
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_datasets.core.dataset._datasets import Dataset

logger = get_logger(__name__)


class ComposedTransform:
    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, sample):
        for transform in self._transforms:
            sample = transform(sample)
        return sample


class DefaultOutputTransformer:
    def __init__(
        self,
        data_dir: str,
        store_artifact_content: bool = True,
        resize_images: bool = False,
        image_max_size: int | tuple[int, int] | None = None,
    ):
        self._data_dir = data_dir
        self._store_artifact_content = store_artifact_content
        self._resize_images = resize_images
        self._image_max_size = image_max_size

    def __call__(self, sample: BaseDataInstance) -> BaseDataInstance:
        if self._store_artifact_content:
            sample = sample.load()
        if (
            self._resize_images
            and isinstance(sample, (ImageInstance, DocumentInstance))
            and sample.image is not None
        ):
            assert self._image_max_size is not None
            if isinstance(self._image_max_size, tuple):
                image = sample.image.ops.resize(
                    width=self._image_max_size[0], height=self._image_max_size[1]
                )
            else:
                image = sample.image.ops.resize_with_aspect_ratio(
                    max_size=self._image_max_size
                )
            sample = sample.update(image=image)
        return sample.ops.convert_file_paths_to_relative(parent_dir=self._data_dir)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_data_dir(data_dir: str | Path) -> str:
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


def _default_data_dir(dataset: Dataset) -> str:
    name = dataset.config.dataset_name or dataset.__class__.__name__
    return str(_DEFAULT_ATRIA_DATASETS_CACHE_DIR / name)



def _get_combined_transform_hash(
    train_transform: DataTransform | None,
    eval_transform: DataTransform | None,
) -> str:
    t = train_transform.hash if train_transform is not None else "none"
    e = eval_transform.hash if eval_transform is not None else "none"
    return hashlib.md5(f"train:{t}|eval:{e}".encode()).hexdigest()[:8]


def _resolve_output_data_model(
    transform: DataTransform | None,
    fallback: type,
) -> type:
    if isinstance(transform, DataTransform):
        try:
            dm = transform.data_model
            if dm is not None:
                return dm
        except (NotImplementedError, AttributeError):
            pass
    return fallback


def _prepare_downloads(
    dataset: Dataset, data_dir: str, access_token: str | None
) -> list[str]:
    from atria_datasets.core.dataset._datasets import Dataset as _Dataset
    from atria_datasets.core.download_manager._download_manager import DownloadManager

    if dataset.__requires_access_token__ and access_token is None:
        logger.warning(
            "access_token must be passed to download this dataset. "
            f"See `{dataset.metadata.homepage}` for instructions to get the access token"
        )

    if dataset._custom_download.__func__ is not _Dataset._custom_download:
        dataset._custom_download(data_dir, access_token)
    else:
        download_dir = Path(data_dir) / _DEFAULT_DOWNLOAD_PATH
        download_dir.mkdir(parents=True, exist_ok=True)
        download_manager = DownloadManager(
            data_dir=Path(data_dir), download_dir=download_dir
        )
        download_urls = dataset._download_urls()
        if download_urls:
            downloaded = download_manager.download_and_extract(
                download_urls,  # type: ignore[arg-type]
                extract=dataset.__extract_downloads__,
                access_token=access_token,
            )
            logger.info(f"Downloaded files {downloaded}")
        return downloaded


def _prepare_split(
    dataset: Dataset,
    split: DatasetSplitType,
    data_dir: str,
    split_iterator_type: type[SplitIterator],
    store_artifact_content: bool = True,
    resize_images: bool = False,
    image_max_size: int | None = None,
    user_transform: DataTransform | None = None,
) -> SplitIterator:
    limits = {
        DatasetSplitType.train: dataset.config.max_train_samples,
        DatasetSplitType.validation: dataset.config.max_validation_samples,
        DatasetSplitType.test: dataset.config.max_test_samples,
    }
    output_transform: DefaultOutputTransformer | ComposedTransform = DefaultOutputTransformer(
        data_dir=data_dir,
        store_artifact_content=store_artifact_content,
        resize_images=resize_images,
        image_max_size=image_max_size,
    )
    if user_transform is not None:
        output_transform = ComposedTransform([output_transform, user_transform])
    return split_iterator_type(
        split=split,
        data_model=dataset.data_model,
        input_transform=dataset._input_transform,
        base_iterator=dataset._split_iterator(split, data_dir),  # type: ignore[arg-type]
        max_len=limits[split],
        output_transform=output_transform,
    )


# ---------------------------------------------------------------------------
# Public builder functions
# ---------------------------------------------------------------------------


def load(
    dataset: Dataset,
    data_dir: str | None = None,
    split: DatasetSplitType | None = None,
    access_token: str | None = None,
    split_iterator_type: type[SplitIterator] = SplitIterator,
) -> Dataset:
    """Populate split iterators in-memory and return the dataset."""
    data_dir = _validate_data_dir(data_dir or _default_data_dir(dataset))
    dataset._data_dir = data_dir
    dataset._downloaded_files = _prepare_downloads(dataset, data_dir, access_token)

    split_iterators: dict[DatasetSplitType, SplitIterator] = {}
    for s in dataset._available_splits():
        if split is not None and s != split:
            continue
        split_iterators[s] = _prepare_split(dataset, s, data_dir, split_iterator_type)

    dataset._split_iterators = split_iterators
    return dataset


def cache(
    dataset: Dataset | CachedDataset,
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
    train_transform: DataTransform | None = None,
    eval_transform: DataTransform | None = None,
) -> CachedDataset:
    """Cache splits to disk and return a file-backed CachedDataset.

    Accepts either a raw Dataset or an already-cached CachedDataset as input,
    making it recursively applicable. When transforms are provided they are
    applied per-split (train_transform for train, eval_transform for all others,
    falling back to train_transform when eval_transform is None). The transform
    hash is baked into the output directory name so different transforms produce
    independent cache locations.
    """
    is_cached = isinstance(dataset, CachedDataset)

    # If no transforms and input is already cached, nothing to do.
    if is_cached and train_transform is None and eval_transform is None:
        return dataset

    if (train_transform is None) != (eval_transform is None):
        raise ValueError("Both train_transform and eval_transform must be provided together.")
    if train_transform is not None and eval_transform is not None:
        if type(train_transform) is not type(eval_transform):
            raise TypeError(
                f"train_transform and eval_transform must be the same type, "
                f"got {type(train_transform).__name__} and {type(eval_transform).__name__}."
            )

    # Resolve storage_dir and the base of unique_config_name.
    # For CachedDataset, storage_dir is always derived from the dataset's own path.
    if is_cached:
        storage_dir = dataset._path.parent
        base_config_name = dataset._path.name
        resolved_data_dir: str | None = None
    else:
        resolved_data_dir = _validate_data_dir(data_dir or _default_data_dir(dataset))
        dataset._data_dir = resolved_data_dir
        storage_dir = Path(resolved_data_dir) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
        base_config_name = dataset.config.config_name + "-" + dataset.config.hash

    unique_config_name = base_config_name
    if train_transform is not None:
        unique_config_name += "-" + _get_combined_transform_hash(train_transform, eval_transform)

    storage_manager = _get_storage_manager(
        cached_storage_type,
        storage_dir=str(storage_dir),
        config_name=unique_config_name,
        num_processes=num_processes,
    )

    # Collect splits that still need writing.
    available_splits = list(dataset.split_iterators) if is_cached else dataset._available_splits()
    splits_to_cache: list[DatasetSplitType] = []
    for s in available_splits:
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

    # Downloads only needed for raw Dataset input.
    if not is_cached and splits_to_cache:
        assert resolved_data_dir is not None
        dataset._downloaded_files = _prepare_downloads(dataset, resolved_data_dir, access_token)  # type: ignore[assignment]

    for s in splits_to_cache:
        logger.info(f"Caching split [{s.value}] to {storage_dir}")
        tf = train_transform if s == DatasetSplitType.train else (eval_transform or train_transform)
        if is_cached:
            assert tf is not None  # guaranteed: train_transform is not None when is_cached
            split_iterator = dataset.split_iterators[s]
            split_iterator.output_transform = tf
        else:
            assert resolved_data_dir is not None
            split_iterator = _prepare_split(
                dataset, s, resolved_data_dir, split_iterator_type,
                store_artifact_content=store_artifact_content,
                resize_images=max_cache_image_size is not None,
                image_max_size=max_cache_image_size,
                user_transform=tf,
            )
        storage_manager.write_split(split_iterator=split_iterator)

    config_dict = dataset.config if is_cached else dataset.config.model_dump()
    metadata_dict = (
        dataset.metadata.model_dump() if dataset.metadata is not None else {}
        if is_cached
        else dataset.metadata.model_dump()
    )
    _save_dataset_info(str(storage_dir), unique_config_name, config_dict, metadata_dict)

    output_data_model = (
        _resolve_output_data_model(train_transform, dataset.data_model)
        if train_transform is not None
        else dataset.data_model
    )
    _save_snapshot(
        storage_dir=storage_dir,
        config_name=unique_config_name,
        data_model=output_data_model,
        storage_type=cached_storage_type,
        dataset_name=dataset.dataset_name if is_cached else dataset.config.dataset_name,
        dataset_class_name=dataset.dataset_class_name if is_cached else dataset.__class__.__name__,
        train_transform=train_transform.to_dict() if train_transform is not None else None,
        eval_transform=eval_transform.to_dict() if eval_transform is not None else None,
    )

    return CachedDataset(path=storage_dir / unique_config_name, allowed_keys=allowed_keys)
