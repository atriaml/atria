---
title: atria_registry
---

# atria_registry

`atria_registry` is the horizontal foundation of the entire Atria framework. Every major component in Atria — datasets, model pipelines, transforms, explainers, optimizers, schedulers — is a **configurable module**: a named class with an attached Pydantic config object that can be looked up by name and instantiated from a config file.

## The core idea

In most frameworks, composing components means writing code:

```python
# ad-hoc composition in PyTorch Lightning
model = timm.create_model("resnet50", pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

In Atria, composition is a config operation:

```yaml
model:
  builder_type: timm
  model_name_or_path: resnet50
optimizer:
  _target_: atria_ml.optimizers.AdamConfig
  lr: 0.001
```

The registry translates names to classes and configs to instances — consistently, across every module in the framework.

## What this module provides

| Class | Role |
|---|---|
| `ModuleConfig` | Pydantic base for every component's configuration |
| `ConfigurableModule` | Base class every registered component inherits from |
| `ModuleRegistry` | Singleton that holds all registry groups |
| `RegistryGroup` | A namespaced store of name → config mappings |

## How components register themselves

Any component that wants to participate in the registry decorates itself:

```python
@DATASETS.register("cifar10/standard")
class Cifar10Dataset(ConfigurableModule):
    class Config(ModuleConfig):
        num_classes: int = 10
    __config__ = Config
```

At import time the config's default values are hashed and stored. Later, `load_dataset_config("cifar10/standard")` retrieves and instantiates the config. `config.build()` produces the live object.

## Registry persistence

`RegistryGroup` persists its store to a SQLite database (`registry.db`) alongside the package. This means a fresh Python process can look up registered modules from disk without needing to re-import every module — useful for large monorepos and CLI workflows.

## Relation to other modules

Every other Atria module creates its own `RegistryGroup` (e.g. `DATASETS`, `MODEL_PIPELINES`, `TRANSFORMS`, `EXPLAINERS`) and registers it in the singleton `ModuleRegistry`. The registry is therefore the contract that makes all modules interoperable.
