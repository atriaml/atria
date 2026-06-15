---
title: Module Registry
---

# Module Registry

The registry system has two layers: `RegistryGroup` (one per domain) and `ModuleRegistry` (the global singleton that holds all groups).

## RegistryGroup

A `RegistryGroup` is a named, namespaced store of `module_path → config` mappings.

```python
DATASETS = RegistryGroup(name="DATASETS", package="atria_datasets")
```

**Registration** — a `@DATASETS.register("cifar10/standard")` decorator:
1. Calls the config's `__init__` to get default values.
2. Hashes the config.
3. Stores `{"hash": ..., "config": ...}` at the path `"cifar10/standard"` in the in-memory store.

**Lookup** — `DATASETS.load_module_config("cifar10/standard")`:
1. Walks the path in the in-memory store (falling back to SQLite if not found).
2. Calls Hydra `instantiate` on the stored config dict to reconstruct the `ModuleConfig`.
3. Returns the live config object ready for `.build()`.

**Hierarchical paths** — paths like `"cifar10/standard"` or `"timm/resnet50/classification"` form a tree in the store, allowing listing all variants of a dataset or model family.

**Persistence** — `RegistryGroup.dump()` writes the in-memory store to a SQLite database (`registry.db`) next to the package. On the next process start, `RegistryGroup.load()` reads from disk. This means the CLI can call `load_dataset_config("cifar10/standard")` without importing every dataset module.

## ModuleRegistry

`ModuleRegistry` is a process-wide singleton that collects all `RegistryGroup` instances across Atria packages:

```python
module_registry = ModuleRegistry()
module_registry.add_registry_group("DATASETS", DATASETS)
module_registry.add_registry_group("MODEL_PIPELINES", MODEL_PIPELINES)
```

Groups are accessible by attribute or by item:

```python
module_registry.DATASETS           # RegistryGroup
module_registry["MODEL_PIPELINES"] # same
```

## Named registry groups in Atria

| Group | Package | Contents |
|---|---|---|
| `DATASETS` | `atria_datasets` | All dataset configs |
| `MODEL_PIPELINES` | `atria_models` | Model pipeline configs (builder + task) |
| `TRANSFORMS` | `atria_transforms` | Data transform configs |
| `EXPLAINERS` | `atria_insights` | Explainer configs |
| `EXPLANATION_PIPELINES` | `atria_insights` | Explanation pipeline configs |
| `OPTIMIZERS` | `atria_ml` | Optimizer configs |
| `SCHEDULERS` | `atria_ml` | LR scheduler configs |

## Comparison to HuggingFace AutoModel

HuggingFace's `AutoModel.from_pretrained("bert-base")` pattern applies only to models. Atria's registry applies the same lookup-by-name pattern uniformly to **every** component in the framework — datasets, transforms, metrics, explainers — enabling fully config-driven pipelines where no component is hardcoded.
