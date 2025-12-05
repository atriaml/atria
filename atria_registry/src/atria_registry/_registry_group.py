"""Registry Group Module"""

from __future__ import annotations

import copy
import importlib
import json
import typing
from pathlib import Path
from typing import Any, Generic

import yaml
from atria_logger import get_logger
from pydantic import BaseModel, ValidationError

from atria_registry._common import T_ModuleConfig
from atria_registry._module_base import (
    ConfigurableModule,
    ModuleConfig,
    RegisterablePydanticModule,
)
from atria_registry._module_builder import ModuleBuilder
from atria_registry._utilities import (
    _extract_nested_defaults,
    _get_config_hash,
    _resolve_module_from_path,
)

logger = get_logger(__name__)


class ConfigSpec(BaseModel):
    hash: str
    config: dict[str, Any]
    import_path: str | None = None


class RegistryGroup(Generic[T_ModuleConfig]):
    __module_builder_class__ = ModuleBuilder
    __exclude_from_builder__ = set()

    def __init__(self, name: str, package: str):
        """
        Initializes the RegistryGroup.

        Args:
            name (str): The name of the registry group.
            package (str): The default provider name for the registry group.
        """

        self._name = name
        self._package = package
        self._store: dict[str, Any] = {}
        self.load()

    @property
    def name(self) -> str:
        """
        Get the name of the registry group.

        Returns:
            str: The name of the registry group.
        """
        return self._name

    @property
    def store(self):
        """
        Get the store for the current registry group.

        Returns:
            dict: The store for the registry group.
        """

        return self._store

    def _package_dir(self) -> Path:
        """Return the filesystem path of the package passed at construction."""
        module = importlib.import_module(self._package)
        assert module is not None, (
            f"Could not find module for package '{self._package}'"
        )
        assert module.__file__ is not None, (
            f"Module '{self._package}' does not have a __file__ attribute."
        )
        return Path(module.__file__).parent

    def register(
        self, module_name: str, configs: dict[str, ModuleConfig | dict] | None = None
    ):
        def decorator(module):
            # first we check if the type of the module is type[ConfigurableModule] if so we directly register it
            if issubclass(module, ConfigurableModule):
                # check if configs are provided, if not register with default config
                if configs is None:
                    default_config = module.__config__()
                    self._register_module(
                        module=module, module_name=module_name, config=default_config
                    )
                else:
                    for config_name, config in configs.items():
                        assert isinstance(config, ModuleConfig), (
                            "Configs must be provided as ModuleConfig for ConfigurableModule."
                        )
                        self._register_module(
                            module=module,
                            module_name=module_name + "/" + config_name,
                            config=config,
                        )
                return module
            elif issubclass(module, RegisterablePydanticModule):
                logger.debug(
                    f"Registering {module=} with {module_name=} and default config in registry group {self._name}."
                )

                # initialized
                if configs is None:
                    # get required kwargs from the module signature
                    config = _extract_nested_defaults(module)

                    logger.debug(
                        f"Registered module at path: {module_name} with config: {config}"
                    )

                    self._register_module(
                        module=module,
                        module_name=module_name,
                        config=config,  # initialize with default
                    )
                elif configs is not None:
                    for config in configs:
                        assert isinstance(config, dict), (
                            "Configs must be provided as dicts."
                        )
                        initialized = module(**config)

                        logger.debug(
                            f"Registered module at path: {module_name} with config: {initialized.model_dump()}"
                        )

                        self._register_module(
                            module=module,
                            module_name=module_name,
                            config=initialized,  # initialize with config
                        )
                return module
            else:  # else we wrap it with ModuleBuilder
                if configs is None:
                    builder = self.__module_builder_class__(module=module)
                    self._register_module(
                        module=builder.__class__,
                        module_name=module_name,
                        config=builder.config,
                    )
                else:
                    for config in configs:
                        assert isinstance(config, ModuleConfig), (
                            "Configs must be provided as ModuleConfig for ConfigurableModule."
                        )
                        builder = self.__module_builder_class__(
                            module=module, **config.model_dump()
                        )
                        self._register_module(
                            module=builder.__class__,
                            module_name=module_name,
                            config=builder.config,
                        )
            return module

        return decorator

    def get_store_value_at_path(self, module_path: str) -> Any:
        cur = self._store
        parts = module_path.strip("/").split("/")

        for d in parts:
            if not isinstance(cur, dict) or d not in cur:
                return None  # or raise KeyError
            cur = cur[d]

        return copy.deepcopy(cur)

    def set_store_value_at_path(self, module_path: str, value: Any) -> None:
        cur = self._store
        parts = module_path.strip("/").split("/")

        for d in parts[:-1]:  # walk through all but the last key
            if d not in cur or not isinstance(cur[d], dict):
                cur[d] = {}
            cur = cur[d]

        # now set value at final key
        cur[parts[-1]] = value

    def _register_module(
        self,
        module: type[ConfigurableModule]
        | type[RegisterablePydanticModule]
        | type[ModuleBuilder],
        module_name: str,
        config: T_ModuleConfig | RegisterablePydanticModule | dict[str, Any],
    ):
        # first make loadable module path
        module_path = (
            module.__module__ + "." + module.__name__
            if module.__module__ != "__main__"
            else module.__name__
        )

        # second get config import path
        if isinstance(config, RegisterablePydanticModule):
            config_import_path = (
                config.__module__ + "." + config.__class__.__name__
                if config.__module__ != "__main__"
                else config.__class__.__name__
            )
        elif isinstance(config, ModuleConfig):
            config_import_path = config.__module__ + "." + config.__class__.__name__
        else:
            config_import_path = None

        # update module path in config
        if isinstance(config, ModuleConfig):
            config = config.model_copy(update={"module_path": module_path})
        elif isinstance(config, dict):
            config["module_path"] = module_path

        # get config hash
        config = config.model_dump() if isinstance(config, BaseModel) else config
        config_hash = _get_config_hash(config)

        cur = self.get_store_value_at_path(module_name)
        if cur is not None:
            assert isinstance(cur, dict), (
                f"Expected dict at path {module_name}, got {type(cur)}"
            )
            if "hash" in cur:
                if config_hash == cur["hash"]:
                    logger.debug(
                        f"Module '{module_name}' with  is already registered. Skipping registration."
                    )
                    return

                raise ValueError(
                    f"Module '{module_name}' with  is already registered with a different configuration."
                )

        self.set_store_value_at_path(
            module_name,
            ConfigSpec(
                hash=config_hash, config=config, import_path=config_import_path
            ).model_dump(),
        )

        # log registration for debugging
        logger.debug(f"Registered module at path: {module_name} with config: {config}")

    def register_config(self, name: str):
        def decorator(config: T_ModuleConfig):
            assert isinstance(config, ModuleConfig), (
                "Configs must be provided as ModuleConfig for ConfigurableModule."
            )
            config_import_path = config.__module__ + "." + config.__class__.__name__

            # update module path in config
            assert config.module_path is not None, "config.module_path must be set."

            # get config hash
            config = config.model_dump() if isinstance(config, BaseModel) else config  # type: ignore
            config_hash = _get_config_hash(config)

            cur = self.get_store_value_at_path(name)
            if cur is not None:
                assert isinstance(cur, dict), (
                    f"Expected dict at path {name}, got {type(cur)}"
                )
                if "hash" in cur:
                    if config_hash == cur["hash"]:
                        logger.debug(
                            f"Module '{name}' with  is already registered. Skipping registration."
                        )
                        return

                    raise ValueError(
                        f"Module '{name}' with  is already registered with a different configuration."
                    )

            self.set_store_value_at_path(
                name,
                ConfigSpec(
                    hash=config_hash, config=config, import_path=config_import_path
                ).model_dump(),
            )

            # log registration for debugging
            logger.debug(f"Registered module at path: {name} with config: {config}")
            return config

        return decorator

    def _validate_non_missing_fields(
        self, module_path: str, config: dict[str, Any], parent_key: str = ""
    ) -> None:
        # go through the config recursively and check if there is any field with value "???" and raise an error
        for key, value in config.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                self._validate_non_missing_fields(module_path, value, current_key)
            elif value == "???":
                raise ValueError(
                    f"Config for module_path={module_path} is missing required field: {current_key}"
                )

    def load_module_config(
        self, module_path: str, **kwargs
    ) -> T_ModuleConfig | dict[str, Any]:
        """Dynamically load all registered modules in the registry group."""
        node = self.get_store_value_at_path(module_path)
        if node is None or node.get("config", None) is None:
            raise KeyError(
                f"Module path '{module_path}' not found in registry. Available paths: {list(self._store.get(self._package, {}).keys())}"
            )
        config_import_path = node.get("import_path", None)
        config = node["config"]
        config.update(kwargs)

        # go through the config recursively and check if there is any field with value "???" and raise an error
        self._validate_non_missing_fields(module_path, config)

        if config_import_path is not None:
            config_cls = typing.cast(
                T_ModuleConfig, _resolve_module_from_path(config_import_path)
            )
            try:
                return config_cls.model_validate(config)
            except ValidationError as e:
                logger.error(
                    f"Error validating config={config_cls} with data={config} for module_path={module_path}"
                )
                raise e
        else:
            return config

    def dump(self, path: Path | None = None) -> Path:
        """Dump registry.json into the package folder."""
        if path is None:
            pkg_dir = self._package_dir()
            path = pkg_dir / "registry.json"

        with open(path, "w") as f:
            logger.debug(f"Dumping '{self._name}' group registry to {path}")
            json.dump(self._store, f, indent=4)

        return path

    def load(self):
        """Load registry.json from the package folder."""
        pkg_dir = self._package_dir()
        path = pkg_dir / "registry.json"

        if not path.exists():
            return

        with open(path) as f:
            logger.debug(f"Loading '{self._name}' group registry from {path}")
            self._store = json.load(f)

    def __repr__(self) -> str:
        return f"<RegistryGroup name={self._name} package={self._package} registered_modules={yaml.dump(self.store, indent=4)}>"

    def __str__(self) -> str:
        return self.__repr__()
