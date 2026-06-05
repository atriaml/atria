"""Registration base classes for Atria modules."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Generic, Self, TypeVar, cast

from atria_logger import get_logger
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict

from atria_registry._utilities import (
    _get_config_hash,
    _resolve_module_from_path,
    to_instantiable_dict,
)

T_ModuleConfig = TypeVar("T_ModuleConfig", bound="ModuleConfig")

logger = get_logger(__name__)


class ModuleConfig(RepresentationMixin, BaseModel):
    """
    Base class for Atria module registry configurations.
    All registry configurations must inherit from this class.
    """

    __version__ = "0.0.0"
    __title__: ClassVar[str] | None = None
    __builds_with_kwargs__ = False
    __hash_exclude__: ClassVar[set[str]] = set()
    __schema_exclude__: ClassVar[set[str]] = set()
    __module_path__: ClassVar[str]

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    @property
    def module_path(self):
        return self.__module_path__

    @property
    def hash(self) -> str:
        config = self.model_dump(exclude=self.__hash_exclude__)
        return _get_config_hash(config)

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        """Create a ModuleConfig from a dict, resolving _target_ entries."""
        from hydra.utils import instantiate

        return instantiate(obj)

    @classmethod
    def __get_pydantic_json_schema__(cls, source, handler):
        schema = handler(source)
        for field in cls.__schema_exclude__:
            schema.get("properties", {}).pop(field, None)

        # update the module_path to be read-only and hidden from the form
        if "module_path" in schema.get("properties", {}):
            schema["properties"]["module_path"]["readOnly"] = True
            schema["properties"]["module_path"]["ui"] = {"hidden": True}

        # update title
        if cls.__title__ is None:
            title = cls.__name__.replace("Config", "")
            # Insert spaces before capital letters
            title = "".join(" " + c if c.isupper() else c for c in title).strip()
            schema["title"] = title
        else:
            schema["title"] = cls.__title__
        return schema

    def to_dict(self) -> dict:
        """Convert the ModuleConfig to a dict suitable for Hydra instantiate."""
        return to_instantiable_dict(self)

    def build(self, **kwargs) -> Any:
        assert self.module_path is not None, (
            "module_path must be set to build the module for config "
            f"{self.__class__.__name__}."
        )
        module = _resolve_module_from_path(self.module_path)
        if isinstance(module, type):
            if self.__builds_with_kwargs__:
                current_kwargs = self.kwargs
                current_kwargs.update(kwargs)
                return module(**current_kwargs)
            else:
                return module(config=self, **kwargs)
        else:
            raise TypeError(
                f"Module at path {self.module_path} is neither a class nor a callable."
            )

    def unsafe_update(self, **kwargs: Any):
        """Return a new ModuleConfig with updated kwargs."""
        for key, value in kwargs.items():
            self.__dict__[key] = value  # type: ignore


class ConfigurableModule(RepresentationMixin, Generic[T_ModuleConfig], ABC):
    """
    Base class for Atria modules that can be registered in the Atria registry.
    All modules that are to be registered must inherit from this class.
    """

    __config__: type[T_ModuleConfig]
    __abstract__: bool = False

    def __init__(self, config: T_ModuleConfig | dict | None = None) -> None:
        if config is None:
            # Use the class's default config
            self._config = self.get_default_config()
        elif isinstance(config, dict):
            # Convert dict to config object
            self._config = self.__config__.model_validate(config)
        else:
            # Already a config object, validate it
            self._config = self.__config__.model_validate(config)

    @classmethod
    def get_default_config(cls) -> T_ModuleConfig:
        """Get default config instance. Override in subclasses if needed."""
        return cast(
            T_ModuleConfig,
            cls.__config__(),
        )

    @classmethod
    def get_config_class(cls) -> type[T_ModuleConfig]:
        """Get the config class for this module."""
        return cast(type[T_ModuleConfig], cls.__config__)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract classes
        if cls.__dict__.get("__abstract__", False):
            return

        if not hasattr(cls, "__config__"):
            raise TypeError(
                f"{cls.__name__} must define a `__config__` class attribute."
            )

        if not issubclass(cls.__config__, ModuleConfig):
            raise TypeError(
                f"{cls.__name__}.__config__ must subclass ModuleConfig. "
                f"Got {cls.__config__} instead."
            )

        path = cls.__module__ + "." + cls.__qualname__
        cls.__config__.__module_path__ = path

    @property
    def config(self) -> T_ModuleConfig:
        return self._config


class PydanticConfigurableModule(RepresentationMixin, BaseModel):
    __version__ = "0.0.0"
    __hash_exclude__: ClassVar[set[str]] = set()
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    @property
    def hash(self) -> str:
        config = self.model_dump(exclude=self.__hash_exclude__)
        return _get_config_hash(config)

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        """Create a ModuleConfig from a dict, resolving _target_ entries."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        omega_conf = OmegaConf.create(obj)
        obj = instantiate(omega_conf)
        return cls.model_validate(obj)

    def to_dict(self) -> dict:
        """Convert the ModuleConfig to a dict suitable for Hydra instantiate."""
        return to_instantiable_dict(self)
