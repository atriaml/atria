"""Dataset builder functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atria_logger import get_logger
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
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_datasets.core.dataset._cached_dataset import CachedDataset
    from atria_datasets.core.dataset._datasets import Dataset

logger = get_logger(__name__)


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
) -> SplitIterator:
    limits = {
        DatasetSplitType.train: dataset.config.max_train_samples,
        DatasetSplitType.validation: dataset.config.max_validation_samples,
        DatasetSplitType.test: dataset.config.max_test_samples,
    }
    return split_iterator_type(
        split=split,
        data_model=dataset.data_model,
        input_transform=dataset._input_transform,
        base_iterator=dataset._split_iterator(split, data_dir),  # type: ignore[arg-type]
        max_len=limits[split],
        output_transform=DefaultOutputTransformer(
            data_dir=data_dir,
            store_artifact_content=store_artifact_content,
            resize_images=resize_images,
            image_max_size=image_max_size,
        ),
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
    dataset: Dataset,
    data_dir: str | None = None,
    split: DatasetSplitType | None = None,
    access_token: str | None = None,
    cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
    overwrite_existing_cached: bool = False,
    store_artifact_content: bool = True,
    max_cache_image_size: int | None = None,
    num_processes: int = 8,
    split_iterator_type: type[SplitIterator] = SplitIterator,
) -> CachedDataset:
    """Cache splits to disk and return a file-backed CachedDataset."""
    from atria_datasets.core.dataset._cached_dataset import CachedDataset

    data_dir = _validate_data_dir(data_dir or _default_data_dir(dataset))
    dataset._data_dir = data_dir
    storage_dir = Path(data_dir) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
    dataset_config = dataset.config
    unique_config_name = dataset_config.config_name + "-" + dataset_config.hash

    storage_manager = _get_storage_manager(
        cached_storage_type,
        storage_dir=str(storage_dir),
        config_name=unique_config_name,
        num_processes=num_processes,
    )

    splits_to_cache: list[DatasetSplitType] = []
    for s in dataset._available_splits():
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
            logger.info(
                f"Loading cached split {s.value} from {storage_manager.split_dir(s)}"
            )

    if splits_to_cache:
        dataset._downloaded_files = _prepare_downloads(dataset, data_dir, access_token)

    info_saved = False
    for s in splits_to_cache:
        logger.info(f"Caching split [{s.value}] to {storage_dir}")
        split_iterator = _prepare_split(
            dataset,
            s,
            data_dir,
            split_iterator_type,
            store_artifact_content=store_artifact_content,
            resize_images=max_cache_image_size is not None,
            image_max_size=max_cache_image_size,
        )
        storage_manager.write_split(split_iterator=split_iterator)
        if not info_saved:
            _save_dataset_info(
                str(storage_dir),
                unique_config_name,
                dataset_config.model_dump(),
                dataset.metadata.model_dump(),
            )
            info_saved = True

    if not info_saved:
        _save_dataset_info(
            str(storage_dir),
            unique_config_name,
            dataset_config.model_dump(),
            dataset.metadata.model_dump(),
        )

    _save_snapshot(
        storage_dir=storage_dir,
        config_name=unique_config_name,
        data_model=dataset.data_model,
        storage_type=cached_storage_type,
        dataset_name=dataset_config.dataset_name,
        dataset_class_name=dataset.__class__.__name__,
    )

    return CachedDataset(path=storage_dir / unique_config_name)
