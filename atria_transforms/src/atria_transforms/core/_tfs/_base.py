"""Base class for data transforms."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from atria_logger import get_logger
from atria_registry._module_base import PydanticConfigurableModule
from pydantic import ConfigDict

from .._data_types import T_TensorDataModel

logger = get_logger(__name__)

T = TypeVar("T")


class DataTransform(PydanticConfigurableModule, Generic[T_TensorDataModel]):
    """Base class for data transforms.
    Transforms should be stateless and operate on input data instances to produce
    transformed output data instances.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    @property
    def data_model(self) -> type[T_TensorDataModel]:
        """Returns the data model class that this transform outputs."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, input: Any) -> T_TensorDataModel | list[T_TensorDataModel]:
        raise NotImplementedError
