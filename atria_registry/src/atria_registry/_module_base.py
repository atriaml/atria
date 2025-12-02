"""Registration base classes for Atria modules."""

import hashlib
import json
from typing import Any, Generic, TypeVar

from atria_logger import get_logger
from atria_types._constants import _MAX_REPR_PRINT_ELEMENTS
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict
from rich.pretty import pretty_repr

T_ModuleConfig = TypeVar("T_ModuleConfig", bound="ModuleConfig")

logger = get_logger(__name__)


class ModuleConfig(BaseModel, RepresentationMixin):
    """
    Base class for Atria module registry configurations.
    All registry configurations must inherit from this class.
    """

    __version__ = "0.0.0"
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str | None = None
    config_name: str = "default"


class RegisterableModule(RepresentationMixin, Generic[T_ModuleConfig]):
    """
    Base class for Atria modules that can be registered in the Atria registry.
    All modules that are to be registered must inherit from this class.
    """

    __config__: type[T_ModuleConfig]

    def __init__(
        self, config: T_ModuleConfig | None = None, **overrides: dict[str, Any]
    ) -> None:
        self._config: T_ModuleConfig = config or self.__config__(**overrides)  # type: ignore

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate presence of Config at class definition time
        if not hasattr(cls, "__config__"):
            raise TypeError(
                f"{cls.__name__} must define a nested `__config__` class inherited from ModuleConfig."
            )

        if not issubclass(cls.__config__, ModuleConfig):
            raise TypeError(
                f"{cls.__name__}.__config__ must subclass ModuleConfig. Got {cls.__config__} instead."
            )

    @property
    def config(self) -> T_ModuleConfig:
        return self._config

    def __repr__(self) -> str:
        config_repr = pretty_repr(
            self._config, max_length=_MAX_REPR_PRINT_ELEMENTS, expand_all=False
        )
        return f"{self.__class__.__name__}(config={config_repr})"

    def __str__(self) -> str:
        return self.__repr__()


class RegisterablePydanticModule(RepresentationMixin, BaseModel):
    """
    Base class for Atria modules that can be registered in the Atria registry.
    All modules that are to be registered must inherit from this class.
    """

    name: str | None = None
    config_name: str = "default"

    def hash(self):
        params = self.model_dump()
        return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[
            :8
        ]
