"""Base class for data transforms."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from atria_logger import get_logger
from atria_registry._module_base import RegisterablePydanticModule
from pydantic import ConfigDict

from .._data_types import T_TensorDataModel

logger = get_logger(__name__)

T = TypeVar("T")


class DataTransform(RegisterablePydanticModule, Generic[T_TensorDataModel]):
    """Base class for data transforms.
    Transforms should be stateless and operate on input data instances to produce
    transformed output data instances.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        original_call = cls.__call__

        def wrapped_call(
            self, input: Any
        ) -> T_TensorDataModel | list[T_TensorDataModel]:
            logger.debug(f"Calling transform: {cls.__name__}")
            result = original_call(self, input)
            logger.debug(f"Transform {cls.__name__} completed")
            return result

        cls.__call__ = wrapped_call

    @abstractmethod
    def __call__(self, input: Any) -> T_TensorDataModel | list[T_TensorDataModel]:
        raise NotImplementedError
