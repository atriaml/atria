---
title: Usage
---

# Usage

## Authentication

```bash
# Register
atria sign_up --username alice --email alice@example.com --password Secret123

# Sign in (stores token in OS keyring)
atria sign_in --email alice@example.com --password Secret123
```

## Dataset operations

```bash
# Prepare a registered dataset and upload to hub
atria datasets prepare_and_upload \
    --name cifar10/standard \
    --target_name alice/cifar10 \
    --data_dir ./data/cache

# Download a hub dataset
atria datasets download --name alice/cifar10 --download_dir ./data/cifar10/
```

## Model operations

```bash
# Upload a model snapshot directory to hub
# (snapshot_dir must contain model.safetensors and metadata.yaml)
atria models upload \
    --name alice/resnet50-cifar10 \
    --snapshot_dir ./runs/exp1/snapshot/

# Download a hub model snapshot
atria models download --name alice/resnet50-cifar10 --download_dir ./models/
```

## Check hub connectivity

```python
from atria_hub.hub import AtriaHub
hub = AtriaHub()
hub.health_check.health_check()
```
