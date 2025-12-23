"""Registration base classes for Atria modules."""

from __future__ import annotations

from abc import ABC
from typing import Any, Generic, Self, TypeVar, cast

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
    __builds_with_kwargs__ = False
    __hash_exclude__: set[str] = set()
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)
    module_path: str | None = None

    @property
    def hash(self) -> str:
        return _get_config_hash(self.model_dump(exclude=self.__hash_exclude__))

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"module_path"})

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

    def to_yaml(self) -> str:
        """Serialize the ModuleConfig to a YAML string."""
        from omegaconf import OmegaConf

        return OmegaConf.to_yaml(OmegaConf.create(self.to_dict()))

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
            cls.__config__(module_path=cls.__module__ + "." + cls.__name__),
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

    @property
    def config(self) -> T_ModuleConfig:
        return self._config


class PydanticConfigurableModule(RepresentationMixin, BaseModel):
    __version__ = "0.0.0"
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    @property
    def hash(self) -> str:
        return _get_config_hash(self.model_dump())

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump()

    def to_yaml(self) -> str:
        """Serialize the ModuleConfig to a YAML string."""
        from omegaconf import OmegaConf

        config_omegaconf = OmegaConf.create(self.model_dump())
        return OmegaConf.to_yaml(config_omegaconf)
