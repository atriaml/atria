---
title: Commands & Config
---

# Commands & Config

## Authentication commands

```bash
# Create a hub account
atria sign_up --username alice --email alice@example.com --password Secret123

# Authenticate (stores credentials in OS keyring)
atria sign_in --email alice@example.com --password Secret123

# Remove stored credentials
atria sign_out
```

`sign_in` uses `email` (not `username`). Credentials are stored via the OS keyring and reused automatically for subsequent hub calls.

## Dataset commands

### prepare_and_upload

Downloads a registered dataset, caches it to Delta Lake format, and pushes it to the hub as a versioned artifact:

```bash
atria datasets prepare_and_upload \
    --name cifar10/standard \
    --target_name my-org/cifar10-custom \
    --data_dir /data/cache \
    --is_public False
```

Key parameters:

| Parameter | Description |
|---|---|
| `name` | Registered dataset name (e.g. `"cifar10/standard"`) |
| `target_name` | Hub artifact name; defaults to the dataset name if omitted |
| `branch` | Hub branch to push to (default: `"main"`) |
| `data_dir` | Local directory for cached storage |
| `is_public` | Whether the artifact is publicly visible |
| `overwrite_existing` | Overwrite if a snapshot already exists on this branch |

Internally: `load_dataset_config(name)` → `config.build(...)` → `dataset.upload_to_hub(name=target_name, ...)`

### download

Pulls a dataset artifact from the hub:

```bash
atria datasets download --name my-org/cifar10-custom --branch main --download_dir ./data/cifar10/
```

## Model commands

### upload

Loads a model snapshot from a local snapshot directory (containing `model.safetensors` and `metadata.yaml`) and uploads it to the hub:

```bash
atria models upload \
    --name my-org/resnet50-cifar10 \
    --snapshot_dir ./runs/exp1/snapshot/ \
    --branch main
```

Key parameters:

| Parameter | Description |
|---|---|
| `name` | Hub artifact name (e.g. `"my-org/resnet50-cifar10"`) |
| `snapshot_dir` | Path to local snapshot directory from `pipeline.save_to_disk()` |
| `branch` | Hub branch to push to (default: `"main"`) |
| `is_public` | Whether the artifact is publicly visible |

### download

Pulls a model snapshot from the hub:

```bash
atria models download --name my-org/resnet50-cifar10 --branch main --download_dir ./models/
```

## Why Fire over Click/Typer

Python Fire was chosen because:
- CLI commands are plain functions — no decorator overhead, no schema definition
- Fire automatically handles nested dicts and lists as CLI arguments
- Adding a new command means adding a new function, not a new decorator chain

The trade-off is less control over help text and argument validation, but for a research tool where the Python API carries the complexity, this is acceptable.
