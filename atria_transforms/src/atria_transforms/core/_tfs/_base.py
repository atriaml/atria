"""Base class for data transforms."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from atria_logger import get_logger
from atria_registry._module_base import PydanticConfigurableModule
from pydantic import ConfigDict

logger = get_logger(__name__)

T = TypeVar("T")


class DataTransform(PydanticConfigurableModule, Generic[T]):
    """Base class for data transforms.
    Transforms should be stateless and operate on input data instances to produce
    transformed output data instances.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="ignore"
    )

    @property
    def data_model(self) -> type[T]:
        """Returns the data model class that this transform outputs."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, input: Any) -> T | list[T]:
        raise NotImplementedError
