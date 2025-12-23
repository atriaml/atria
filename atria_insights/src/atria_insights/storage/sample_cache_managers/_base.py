from abc import abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger

from atria_insights.storage.data_cachers._common import CacheData
from atria_insights.storage.data_cachers._hdf5 import HDF5DataCacher

logger = get_logger(__name__)

T = TypeVar("T")


class BaseSampleCacheManager(Generic[T]):
    def __init__(
        self, cache_dir: str | Path, file_name: str, cacher_type: str = "hdf5"
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if cacher_type == "hdf5":
            self._cacher = HDF5DataCacher(file_path=str(self._cache_dir / file_name))
        else:
            raise ValueError(f"Unsupported cacher_type: {cacher_type}")

    @property
    def file_path(self) -> Path:
        return self._cacher._file_path

    def sample_exists(self, sample_key: str) -> bool:
        return self._cacher.sample_exists(sample_key)

    def save_sample(self, data: T) -> None:
        cached_data = self._to_cache_data(data)
        self._cacher.save_sample(cached_data)

    def load_sample(self, sample_key: str) -> T:
        cache_data = self._cacher.load_sample(sample_key)
        return self._from_cache_data(cache_data)

    def list_sample_keys(self) -> list[str]:
        return self._cacher.list_sample_keys()

    @abstractmethod
    def _to_cache_data(self, data: T) -> CacheData: ...

    @abstractmethod
    def _from_cache_data(self, data: CacheData) -> T: ...
