from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger

from atria_insights.storage.data_cachers._common import CacheData

logger = get_logger(__name__)

T = TypeVar("T")


class DataCacher(ABC):
    def __init__(self, file_path: str):
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def sample_exists(self, sample_key: str) -> bool: ...

    @abstractmethod
    def list_sample_keys(self) -> list[str]: ...

    @abstractmethod
    def save_sample(self, data: CacheData) -> None: ...

    @abstractmethod
    def load_sample(self, sample_key: str) -> CacheData: ...
