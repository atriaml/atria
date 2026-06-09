from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger

from atria_insights.storage.data_cachers._base import DataCacher
from atria_insights.storage.data_cachers._common import SerializableSampleData

logger = get_logger(__name__)

T = TypeVar("T")


class BaseSampleCacheManager(Generic[T]):
    def __init__(self, cacher: DataCacher):
        self._cacher = cacher

    @property
    def file_path(self) -> Path:
        return self._cacher._file_path

    def save_file_attrs(self, attrs: dict[str, Any]) -> None:
        self._cacher.save_file_attrs(attrs)

    def sample_exists(self, sample_key: str) -> bool:
        return self._cacher.sample_exists(sample_key)

    def save_sample(self, data: T) -> None:
        cached_data = self._serialize_type(data)
        self._cacher.save_sample(cached_data)

    def load_sample(self, sample_key: str, load_tensors: bool = True) -> T:
        cache_data = self._cacher.load_sample(sample_key, load_tensors=load_tensors)
        return self._deserialize_type(cache_data)

    def load_sample_attrs(self, sample_key: str) -> dict[str, Any]:
        return self._cacher.load_sample_attrs(sample_key)

    def list_sample_keys(self) -> list[str]:
        return self._cacher.list_sample_keys()

    @abstractmethod
    def _serialize_type(self, data: T) -> SerializableSampleData: ...

    @abstractmethod
    def _deserialize_type(self, data: SerializableSampleData) -> T: ...
