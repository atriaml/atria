"""Private helpers for dataset loading and caching."""

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
from atria_datasets.core.dataset._split_iterators import SplitIterator

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


class PreprocessOutputTransformer:
    """Output transformer for the cache-write step.

    Loads artifact content, optionally resizes images, and converts absolute
    file paths to relative ones so the cached snapshot is portable.
    """

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
        sample = sample.ops.convert_file_paths_to_relative(parent_dir=self._data_dir)
        return sample


class LoadOutputTransformer:
    """Output transformer for runtime loading (cached or uncached).

    Calls sample.load() to resolve any stored file references into content,
    then optionally composes with a user-supplied runtime transform.
    """

    def __call__(self, sample: BaseDataInstance) -> BaseDataInstance:
        return sample.load()


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
    preprocess_train_transform: DataTransform | None,
    preprocess_eval_transform: DataTransform | None,
) -> str:
    t = (
        preprocess_train_transform.hash
        if preprocess_train_transform is not None
        else "none"
    )
    e = (
        preprocess_eval_transform.hash
        if preprocess_eval_transform is not None
        else "none"
    )
    return hashlib.md5(f"train:{t}|eval:{e}".encode()).hexdigest()[:8]


def _resolve_output_data_model(transform: DataTransform | None, fallback: type) -> type:
    if isinstance(transform, DataTransform):
        try:
            dm = transform.data_model
            if dm is not None:
                return dm
        except (NotImplementedError, AttributeError):
            pass
    return fallback


def _compute_unique_cache_path(
    dataset: Dataset,
    data_dir: str | None,
    preprocess_train_transform: DataTransform | None,
    preprocess_eval_transform: DataTransform | None,
) -> Path:
    resolved = _validate_data_dir(data_dir or _default_data_dir(dataset))
    storage_dir = Path(resolved) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
    config_name = dataset.config.config_name + "-" + dataset.config.hash
    if preprocess_train_transform is not None:
        config_name += "-" + _get_combined_transform_hash(
            preprocess_train_transform, preprocess_eval_transform
        )
    return storage_dir / config_name


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
    for_cache: bool = False,
) -> SplitIterator:
    """Build a SplitIterator for a dataset split.

    for_cache=True  → PreprocessOutputTransformer (load, resize, relativize paths) + user_transform
    for_cache=False → LoadOutputTransformer (sample.load()) + user_transform
    """
    limits = {
        DatasetSplitType.train: dataset.config.max_train_samples,
        DatasetSplitType.validation: dataset.config.max_validation_samples,
        DatasetSplitType.test: dataset.config.max_test_samples,
    }
    if for_cache:
        base_tf: PreprocessOutputTransformer | LoadOutputTransformer = (
            PreprocessOutputTransformer(
                data_dir=data_dir,
                store_artifact_content=store_artifact_content,
                resize_images=resize_images,
                image_max_size=image_max_size,
            )
        )
    else:
        base_tf = LoadOutputTransformer()
    output_transform: (
        PreprocessOutputTransformer | LoadOutputTransformer | ComposedTransform
    ) = base_tf
    if user_transform is not None:
        output_transform = ComposedTransform([base_tf, user_transform])
    return split_iterator_type(
        split=split,
        data_model=dataset.data_model,
        input_transform=dataset.input_transform,
        base_iterator=dataset._split_iterator(split, data_dir),  # type: ignore[arg-type]
        max_len=limits[split],
        output_transform=output_transform,
    )
