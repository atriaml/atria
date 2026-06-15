---
title: Snapshots & Hub
---

# Snapshots & Hub

A **snapshot** is a versioned, self-contained bundle of a `ModelPipeline` checkpoint together with its full configuration. It is the artifact that moves between training, evaluation, explanation, and the hub.

## What a snapshot contains

```python
@dataclass
class SnapshotArtifact:
    weights: bytes    # safetensors.torch.save(state_dict) → bytes
    metadata: bytes   # YAML dump of pipeline_name + config + labels
```

The `weights` are the model `state_dict` serialized with [safetensors](https://github.com/huggingface/safetensors). The `metadata` is a YAML document containing the `pipeline_name`, the complete `ModelPipelineConfig` in Hydra-instantiable dict format, and the `DatasetLabels` — enough to reconstruct the full pipeline without any other context.

## Creating a snapshot

```python
artifact = pipeline.snapshot()
# artifact.weights  → safetensors bytes
# artifact.metadata → YAML bytes (pipeline_name, config, labels)
```

To save to disk:

```python
snapshot_dir = pipeline.save_to_disk("/runs/exp1/")
```

To reconstruct from disk:

```python
pipeline = ModelPipeline.load_from_disk("/runs/exp1/")
```

## Publishing to the hub

```python
pipeline.upload_to_hub(
    name="my-org/resnet50-cifar10",
    branch="main",
    is_public=False,
)
```

On the hub, the snapshot is stored as a versioned model artifact with the weights file and the YAML metadata.

## Pulling from the hub

```python
pipeline = ModelPipeline.load_from_hub(
    name="my-org/resnet50-cifar10",
    branch="main",
)
```

Or using the CLI:

```bash
atria models download my-org/resnet50-cifar10 --output-dir ./models/
```

## Why snapshot + hub instead of just saving weights

In typical workflows, a checkpoint file and the code that produced it quickly diverge. Atria snapshots bundle the full `ModelPipelineConfig` with the weights, ensuring that:
1. The same architecture is always reconstructed from the same snapshot.
2. Hub-published snapshots are reproducible: pulling a snapshot gives back a pipeline that behaves identically to when it was snapshotted.
3. Evaluation and explanation configs can reference a hub snapshot by name rather than a local path.
