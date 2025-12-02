"""Registry Group Module"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Generic

from atria_logger import get_logger
from pydantic import BaseModel

from atria_registry._common import T_RegisterableModule
from atria_registry._module_base import (
    ModuleConfig,
    RegisterableModule,
    RegisterablePydanticModule,
)
from atria_registry._module_builder import ModuleBuilder
from atria_registry._utilities import _extract_nested_defaults, _get_config_hash

logger = get_logger(__name__)


class ModuleSpec(BaseModel):
    module: str
    hash: str
    config: dict[str, Any]


class RegistryGroup(Generic[T_RegisterableModule]):
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
        self, module_name: str, configs: list[ModuleConfig | dict] | None = None
    ):
        def decorator(module):
            # first we check if the type of the module is type[RegisterableModule] if so we directly register it
            if issubclass(module, RegisterableModule):
                # check if configs are provided, if not register with default config
                if configs is None:
                    default_config = module.__config__()
                    self._register_module(
                        module=module, module_name=module_name, config=default_config
                    )
                else:
                    for config in configs:
                        assert isinstance(config, ModuleConfig), (
                            "Configs must be provided as ModuleConfig for RegisterableModule."
                        )
                        self._register_module(
                            module=module, module_name=module_name, config=config
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
                            "Configs must be provided as ModuleConfig for RegisterableModule."
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

    def _register_module(
        self,
        module: type[RegisterableModule]
        | type[RegisterablePydanticModule]
        | type[ModuleBuilder],
        module_name: str,
        config: ModuleConfig | RegisterablePydanticModule | dict[str, Any],
    ):
        logger.debug(
            f"Registering {module=} with {module_name=} and {config=} in registry group {self._name}."
        )

        # determine package and provider
        module_path = f"{self._package}/"

        cur = self._store
        if module_name is not None:
            for d in module_name.split("/"):
                module_path += f"{d}/"
                if d not in cur:
                    cur[d] = {}
                cur = cur[d]

        assert isinstance(cur, dict)

        config = config.model_dump() if isinstance(config, BaseModel) else config
        config_name = config["config_name"]
        module_path += config["config_name"]
        config_hash = _get_config_hash(config)
        if config_name in cur:
            if config_hash == cur[config_name]["hash"]:
                logger.debug(
                    f"Module '{module_path}' with config name '{config_name}' is already registered. Skipping registration."
                )
                return

            raise ValueError(
                f"Module '{module_path}' with config name '{config_name}' is already registered with a different configuration."
            )

        cur[config_name] = ModuleSpec(
            module=module.__module__ + "." + module.__name__
            if module.__module__ != "__main__"
            else module.__name__,
            hash=config_hash,
            config=config,
        ).model_dump()

        # log registration for debugging
        logger.debug(f"Registered module at path: {module_path} with config: {config}")

    def _load_module_and_config(self, module_path: str) -> tuple[Any, dict[str, Any]]:
        """Dynamically load all registered modules in the registry group."""
        cur = self._store
        parent_key = None
        try:
            for cur_key in module_path.split("/"):
                cur = cur[cur_key]
                parent_key = cur_key
        except KeyError:
            available_paths = []
            for cur_key in cur.keys():
                available_paths.append(
                    f"{parent_key}/{cur_key}" if parent_key else cur_key
                )
            raise KeyError(
                f"Module path '{module_path}' not found in registry. Available modules: {available_paths}"
            )

        node = cur
        module = node.get("module", None)
        config = node.get("config", None)
        assert module is not None, f"No module found at path: {module_path}"
        assert isinstance(config, dict), f"No config found at path: {module_path}"

        # import the module
        module_path, class_name = module.rsplit(".", 1)
        imported_module = importlib.import_module(module_path)
        module = getattr(imported_module, class_name)
        return module, config

    def load_module(self, module_path: str, **kwargs) -> Any:
        """Dynamically load all registered modules in the registry group."""
        module, config = self._load_module_and_config(module_path)

        # we need to check if the module is a ModuleBuilder or a RegisterableModule
        # regenerate the config
        if issubclass(module, ModuleBuilder):
            builder = module(**config, **kwargs)
            initialized = builder()
            config = builder.config
            return initialized
        elif issubclass(module, RegisterablePydanticModule):
            initialized = module(**config, **kwargs)
            return initialized  # config is the instance itself
        elif issubclass(module, RegisterableModule):
            config = module.__config__(**config)
            initialized = module(config=config, **kwargs)
            return initialized
        else:
            raise TypeError(
                f"Loaded module '{module.__name__}' is not a valid RegisterableModule or ModuleBuilder."
            )

    def load_module_config(
        self, module_path: str, init_module: bool = True, **kwargs
    ) -> BaseModel | ModuleConfig:
        """Dynamically load all registered modules in the registry group."""
        module, config = self._load_module_and_config(module_path, **kwargs)

        # we need to check if the module is a ModuleBuilder or a RegisterableModule
        # regenerate the config
        if issubclass(module, ModuleBuilder):
            builder = module(**config, **kwargs)
            config = builder.config
            return config
        elif issubclass(module, RegisterablePydanticModule):
            initialized = module(**config, **kwargs)
            return initialized  # config is the instance itself
        elif issubclass(module, RegisterableModule):
            config = module.__config__(**config)
            return config
        else:
            raise TypeError(
                f"Loaded module '{module.__name__}' is not a valid RegisterableModule or ModuleBuilder."
            )

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
        return f"<RegistryGroup name={self._name} package={self._package} registered_modules={len(self.list())}>"

    def __str__(self) -> str:
        return self.__repr__()
