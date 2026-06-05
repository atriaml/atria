"""Registry Group Module"""

from __future__ import annotations

import copy
import importlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Generic

from atria_logger import get_logger
from pydantic import BaseModel

from atria_registry._common import T_ModuleConfig
from atria_registry._module_base import (
    ConfigurableModule,
    ModuleConfig,
    PydanticConfigurableModule,
)

logger = get_logger(__name__)

_BUILD_REGISTRY = os.environ.get("ATRIA_BUILD_REGISTRY", "true").lower() == "true"


class ConfigSpec(BaseModel):
    hash: str
    config: dict[str, Any]


class RegistryGroup(Generic[T_ModuleConfig]):
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

    def list_all_modules(self) -> list[str]:
        """List all registered module paths in the registry group."""
        module_names: list[str] = []

        def _traverse(store: dict[str, Any], prefix: str = "") -> None:
            for key, value in store.items():
                current_path = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict) and "config" in value:
                    module_names.append(current_path)
                else:
                    _traverse(value, current_path)

        _traverse(self._store)
        in_memory = set(module_names)

        try:
            conn = self._get_db_connection()
            rows = conn.execute(
                "SELECT path FROM registry WHERE group_name=?", (self._name,)
            ).fetchall()
            conn.close()
            for (path,) in rows:
                if path not in in_memory:
                    module_names.append(path)
        except Exception as e:
            logger.warning(f"SQLite registry list failed: {e}")

        return module_names

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

    def _db_path(self) -> Path:
        return self._package_dir() / "registry.db"

    def _get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path()))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS registry (
                group_name TEXT NOT NULL,
                path       TEXT NOT NULL,
                hash       TEXT NOT NULL,
                config     TEXT NOT NULL,
                PRIMARY KEY (group_name, path)
            )
        """)
        conn.commit()
        return conn

    def register(
        self, module_name: str, configs: dict[str, ModuleConfig | dict] | None = None
    ):
        if not _BUILD_REGISTRY:

            def noop(module):
                return module

            return noop

        logger.debug(
            f"Registering module '{module_name}' with configs: {configs} in registry group '{self._name}'"
        )

        def decorator(module):
            if issubclass(module, ModuleConfig | PydanticConfigurableModule):
                assert configs is None, (
                    "Configs should not be provided when registering a ModuleConfig subclass."
                )
                # initialize the config to default values
                config = module()
                config_hash = config.hash
                config = config.to_dict()

                cur = self.get_store_value_at_path(module_name, load_from_db=False)
                if cur is not None:
                    assert isinstance(cur, dict), (
                        f"Expected dict at path {module_name}, got {type(cur)}"
                    )
                    if "hash" in cur:
                        if config_hash == cur["hash"]:
                            logger.debug(
                                f"Module '{module_name}' with hash '{config_hash}' is already registered. Skipping registration."
                            )
                            return module

                        logger.warning(
                            f"Module '{module_name}' with hash '{config_hash}' is already registered with a different configuration. Replacing it."
                        )

                self.set_store_value_at_path(
                    module_name,
                    ConfigSpec(hash=config_hash, config=config).model_dump(),
                )

                # log registration for debugging
                logger.debug(
                    f"Registered module at path: {module_name} with config: {config}"
                )
                return module
            elif issubclass(module, ConfigurableModule):
                # check if configs are provided, if not register with default config
                if configs is None:
                    default_config = module.get_default_config()
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
            else:
                raise NotImplementedError(
                    "Only ModuleConfig and ConfigurableModule can be registered."
                )

        return decorator

    def get_store_value_at_path(
        self, module_path: str, load_from_db: bool = True
    ) -> Any:
        # Check in-memory store first
        cur = self._store
        parts = module_path.strip("/").split("/")
        found = True
        for d in parts:
            if not isinstance(cur, dict) or d not in cur:
                found = False
                break
            cur = cur[d]
        if found:
            return copy.deepcopy(cur)

        if load_from_db:
            path_key = "/".join(parts)
            try:
                conn = self._get_db_connection()
                row = conn.execute(
                    "SELECT hash, config FROM registry WHERE group_name=? AND path=?",
                    (self._name, path_key),
                ).fetchone()
                conn.close()
                if row:
                    return {"hash": row[0], "config": json.loads(row[1])}
            except Exception as e:
                logger.warning(f"SQLite registry query failed: {e}")

        return None

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
        module: type[ConfigurableModule],
        module_name: str,
        config: T_ModuleConfig | dict[str, Any],
    ):
        # get config hash
        config_hash = config.hash
        config = config.to_dict()

        cur = self.get_store_value_at_path(module_name, load_from_db=False)
        if cur is not None:
            assert isinstance(cur, dict), (
                f"Expected dict at path {module_name}, got {type(cur)}"
            )
            if "hash" in cur:
                if config_hash == cur["hash"]:
                    logger.debug(
                        f"Module '{module_name}' with hash '{config_hash}' is already registered. Skipping registration."
                    )
                    return

                logger.warning(
                    f"Module '{module_name}' with hash '{config_hash}' is already registered with a different configuration. Replacing it."
                )

        self.set_store_value_at_path(
            module_name, ConfigSpec(hash=config_hash, config=config).model_dump()
        )

        # log registration for debugging
        logger.debug(f"Registered module at path: {module_name} with config: {config}")

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
            all_modules_str = json.dumps(self.list_all_modules(), indent=4)
            raise RuntimeError(
                f"Module path '{module_path}' not found in registry. Available paths:\n {all_modules_str}"
            )
        config = node["config"]

        assert "_target_" in config, (
            f"Config for module_path={module_path} must contain '_target_' field for instantiation."
        )
        from hydra.utils import instantiate

        config.update(kwargs)
        obj = instantiate(config)
        return obj

    def dump(self, path: Path | None = None, refresh: bool = False, to_json: bool = False) -> Path:
        """Dump the in-memory store into the SQLite registry database (or JSON if to_json=True)."""
        if to_json:
            return self._dump_json(path, refresh)
        return self._dump_sqlite(path, refresh)

    def _dump_sqlite(self, path: Path | None = None, refresh: bool = False) -> Path:
        """Dump the in-memory store into the SQLite registry database."""
        db_path = path or self._db_path()
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS registry (
                group_name TEXT NOT NULL,
                path       TEXT NOT NULL,
                hash       TEXT NOT NULL,
                config     TEXT NOT NULL,
                PRIMARY KEY (group_name, path)
            )
        """)
        if refresh:
            conn.execute("DELETE FROM registry WHERE group_name=?", (self._name,))
        def _flatten(d: dict, prefix: str = "") -> None:
            for key, value in d.items():
                p = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict) and "config" in value and "hash" in value:
                    conn.execute(
                        "INSERT OR REPLACE INTO registry VALUES (?,?,?,?)",
                        (self._name, p, value["hash"], json.dumps(value["config"])),
                    )
                elif isinstance(value, dict):
                    _flatten(value, p)
        _flatten(self._store)
        conn.commit()
        conn.close()
        logger.debug(f"Dumped '{self._name}' group registry to {db_path}")
        logger.info(
            f"Registry dump complete. Registered modules:\n{self.list_all_modules()}"
        )
        return db_path

    def _dump_json(self, path: Path | None = None, refresh: bool = False) -> Path:
        """Dump the in-memory store into a JSON file."""
        json_path = path or self._db_path().with_suffix(".json")

        existing: dict = {}
        if not refresh and json_path.exists():
            with open(json_path, "r") as f:
                existing = json.load(f)

        entries = existing.get(self._name, {})

        def _flatten(d: dict, prefix: str = "") -> None:
            for key, value in d.items():
                p = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict) and "config" in value and "hash" in value:
                    entries[p] = {"hash": value["hash"], "config": value["config"]}
                elif isinstance(value, dict):
                    _flatten(value, p)

        _flatten(self._store)
        existing[self._name] = entries

        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.debug(f"Dumped '{self._name}' group registry to {json_path}")
        logger.info(
            f"Registry dump complete. Registered modules:\n{self.list_all_modules()}"
        )
        return json_path

    def _schema_db_path(self) -> Path:
        return self._package_dir() / "schema.db"

    def dump_schema(
        self, path: Path | None = None, refresh: bool = False, to_json: bool = False
    ) -> Path:
        """Dump JSON schemas for all registered configs into a schema.db SQLite database."""
        if to_json:
            json_path = Path("tree.json")

            schema = {}
            if json_path.exists():
                with open(json_path, "rb") as f:
                    schema = json.load(f)

            if refresh or self._name not in schema:
                schema[self._name] = {}
            for module_path in self.list_all_modules():
                cfg = self.load_module_config(module_path)
                cfg = cfg.model_copy(update={"explainability_metrics": None})
                if isinstance(cfg, ModuleConfig):
                    form_schema = cfg.model_json_schema()
                    schema[self._name][module_path] = form_schema
            with open(json_path, "w") as f:
                json.dump(schema, f, indent=4)
            logger.info(f"Schema dump complete for '{self._name}' → {json_path}")
            return json_path
        else:
            db_path = path or self._schema_db_path()
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema (
                    group_name TEXT NOT NULL,
                    path       TEXT NOT NULL,
                    schema     TEXT NOT NULL,
                    PRIMARY KEY (group_name, path)
                )
            """)
            if refresh:
                conn.execute("DELETE FROM schema WHERE group_name=?", (self._name,))

            for module_path in self.list_all_modules():
                try:
                    cfg = self.load_module_config(module_path)
                    if isinstance(cfg, ModuleConfig):
                        form_schema = cfg.model_json_schema()
                        form_schema = json.dumps(form_schema)
                        conn.execute(
                            "INSERT OR REPLACE INTO schema VALUES (?,?,?)",
                            (self._name, module_path, form_schema),
                        )
                except Exception as e:
                    logger.warning(f"Skipping schema for '{module_path}': {e}")

            conn.commit()
            conn.close()
            logger.info(f"Schema dump complete for '{self._name}' → {db_path}")
            return db_path

    def load(self) -> None:
        """Load all entries from SQLite into the in-memory store."""
        db_path = self._db_path()
        if not db_path.exists():
            return
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT path, hash, config FROM registry WHERE group_name=?", (self._name,)
        ).fetchall()
        conn.close()
        for path, hash_val, config_str in rows:
            self.set_store_value_at_path(
                path, {"hash": hash_val, "config": json.loads(config_str)}
            )
        logger.debug(f"Loaded '{self._name}' group registry from {db_path}")

    def __repr__(self) -> str:
        return f"<RegistryGroup name={self._name} package={self._package} store_keys={list(self._store.keys())}>"

    def __str__(self) -> str:
        return self.__repr__()
