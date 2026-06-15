---
title: Publishable Entities
---

# Publishable Entities

A **publishable entity** in Atria is any artifact that carries enough information to be reconstructed by a consumer from the hub alone — without access to the original training code or local filesystem.

## What makes an artifact publishable

Two things are required:

1. **Content** — the raw binary (weights bytes, storage shards, etc.)
2. **Config / metadata** — a serializable Hydra-instantiable dict that fully describes how to reconstruct the artifact

For model snapshots, this is the `SnapshotArtifact.weights` + `SnapshotArtifact.metadata` pair (see [Snapshots & Hub](../../atria_models/concepts/snapshots.md)).

For datasets, this is the dataset's storage shards + the serialized `DatasetConfig` (including preprocessing config, split info, and schema).

## Model artifacts on the hub

A model artifact on the hub is identified by `{owner}/{model-name}` and optionally a version or commit hash. It contains:

- `model_weights.pt` — PyTorch state dict bytes
- `model_config.json` — the full `ModelPipelineConfig` in Hydra format
- Metadata: training dataset name, evaluation metrics, Atria version

Pulling a model artifact restores the exact `ModelPipeline` that was snapshotted:

```python
config = load_model_pipeline_config("my-org/resnet50-cifar10", from_hub=True)
pipeline = config.build(labels=dataset.labels)
# pipeline is identical to what was trained
```

## Dataset artifacts on the hub

A dataset artifact contains:
- The cached storage shards (Deltalake Parquet or msgpack)
- The `DatasetConfig` JSON
- Split info (train/val/test sizes, random seed)

Pulling a dataset artifact rehydrates a `Dataset` object identical to the one that was pushed:

```bash
atria datasets download my-org/cifar10-custom --output-dir ./data/
```

## Versioning

Every push creates a new version. Previous versions remain accessible. The hub tracks:
- Version number (monotonically increasing)
- Commit hash (content-addressed)
- Timestamp and pusher identity

This ensures that pulling a model or dataset by version always returns the same artifact, regardless of subsequent pushes.
