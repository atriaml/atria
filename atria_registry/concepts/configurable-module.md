---
title: Configurable Modules
---

# Configurable Modules

A **configurable module** is the fundamental unit of Atria: a class whose behavior is fully described by an attached Pydantic config object, so it can be constructed from a config alone.

## ModuleConfig

`ModuleConfig` is the base class for every configuration in Atria. It is a frozen Pydantic `BaseModel` with a few extras:

- **`_target_` injection** — the JSON schema automatically includes a `_target_` field pointing to the config's class path. This lets Hydra's `instantiate` reconstruct any config from a plain dict.
- **`hash`** — a deterministic hash of the config's fields, used by the registry to detect duplicates and cache results.
- **`build(**kwargs)`** — resolves `__module_path__` to the owning `ConfigurableModule` class and constructs it with `config=self`.
- **`to_dict()` / `from_dict()`** — round-trip to/from Hydra-instantiable dicts for YAML/JSON serialization.

```python
class MyTransformConfig(ModuleConfig):
    kernel_size: int = 3
    sigma: float = 1.0
```

## ConfigurableModule

`ConfigurableModule` is the base class every registered component inherits from. It is generic over its config type:

```python
class MyTransform(ConfigurableModule[MyTransformConfig]):
    __config__ = MyTransformConfig

    def __call__(self, x):
        ...
```

**Contract enforced at class definition time:**
- Every non-abstract subclass must declare `__config__`.
- `__config__` must be a `ModuleConfig` subclass.
- `__config__.__module_path__` is automatically set to the fully-qualified class path of the owning module.

This means the config always knows which class to build, and the class always knows its config type — no separate registration step required.

## PydanticConfigurableModule

For components that are themselves Pydantic models (rather than holding a config object), `PydanticConfigurableModule` provides the same `hash`, `to_dict`, and `from_dict` API without the `ConfigurableModule` base class overhead. Used internally for data pipeline and storage configurations.

## Lifecycle

```
config dict (YAML/JSON)
    ↓  ModuleConfig.from_dict()
ModuleConfig instance
    ↓  config.build()  OR  RegistryGroup.load_module_config()
ConfigurableModule instance
```

This lifecycle is the same for datasets, models, transforms, explainers, optimizers — every component in the framework.
