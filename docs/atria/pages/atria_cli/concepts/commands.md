---
title: Commands & Config
---

# Commands & Config

## Authentication commands

```bash
# Create a hub account
atria sign_up --username alice --email alice@example.com --password secret

# Authenticate (stores credentials in OS keyring)
atria sign_in --username alice --password secret

# Remove stored credentials
atria sign_out
```

## Dataset commands

### prepare_and_upload

Downloads a raw dataset, converts it to typed `BaseDataInstance` objects, caches it to a storage backend, and pushes it to the hub as a versioned artifact:

```bash
atria datasets prepare_and_upload \
    --dataset-name cifar10/standard \
    --data-dir /data/cache \
    --storage-type MSGPACK \
    --hub-dataset-name my-org/cifar10-custom
```

Internally this runs the full dataset lifecycle:
`load_dataset_config` → `config.build()` → `dataset.push_to_hub()`

### download

Pulls a dataset artifact from the hub and writes storage files to a local directory:

```bash
atria datasets download my-org/cifar10-custom --output-dir ./data/cifar10/
```

## Model commands

### upload

Pushes a model snapshot (weights + config) to the hub:

```bash
atria models upload \
    --checkpoint ./runs/exp1/best.pt \
    --config ./runs/exp1/config.json \
    --hub-model-name my-org/resnet50-cifar10
```

### download

Pulls a model snapshot from the hub:

```bash
atria models download my-org/resnet50-cifar10 --output-dir ./models/
```

## Config file format

All training / evaluation / explanation commands accept a YAML or JSON config file that maps to the appropriate `TaskConfig` subclass. The config is loaded via Hydra `instantiate`, so nested `_target_` fields are supported:

```yaml
# train_config.yaml
_target_: atria_ml.configs.TrainingTaskConfig
env:
  _target_: atria_ml.configs._env.RuntimeEnvConfig
  run_dir: ./runs/exp1
data:
  _target_: atria_ml.configs.DataConfig
  dataset_config:
    _target_: atria_datasets.registry.Cifar10Config
  train_batch_size: 64
model_pipeline:
  _target_: atria_models...ImageClassificationConfig
  model:
    builder_type: timm
    model_name_or_path: resnet50
trainer:
  _target_: atria_ml.configs.TrainerConfig
  learning_rate: 0.001
  max_epochs: 50
```

```bash
atria train --config train_config.yaml
```

## Why Fire over Click/Typer

Python Fire was chosen because:
- CLI commands are plain functions — no decorator overhead, no schema definition
- Fire automatically handles nested dicts and lists as CLI arguments
- Adding a new command means adding a new function, not a new decorator chain

The trade-off is less control over help text and argument validation, but for a research tool where the config file carries the complexity, this is acceptable.
