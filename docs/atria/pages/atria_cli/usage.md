---
title: Usage
---

# Usage

## Authentication

```bash
# Register and sign in
atria sign_up --username alice --email alice@example.com
atria sign_in --username alice --password secret
```

## Dataset operations

```bash
# Prepare a registered dataset and upload to hub
atria datasets prepare_and_upload \
    --dataset-name cifar10/standard \
    --data-dir ./data/cache \
    --hub-dataset-name alice/cifar10

# Download a hub dataset
atria datasets download alice/cifar10 --output-dir ./data/cifar10/
```

## Model operations

```bash
# Upload a trained model snapshot
atria models upload \
    --checkpoint ./runs/exp1/best.pt \
    --config ./runs/exp1/config.json \
    --hub-model-name alice/resnet50-cifar10

# Download a hub model
atria models download alice/resnet50-cifar10 --output-dir ./models/
```

## Training and evaluation (via task configs)

```bash
# Train
python -m atria_cli train --config configs/train.yaml

# Evaluate
python -m atria_cli evaluate --config configs/eval.yaml

# Explain
python -m atria_cli explain --config configs/explain.yaml
```

## Check hub connectivity

```python
from atria_hub.hub import AtriaHub
hub = AtriaHub()
hub._health_check_api.ping()
```
