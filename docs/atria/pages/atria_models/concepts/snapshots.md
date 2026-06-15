---
title: Snapshots & Hub
---

# Snapshots & Hub

A **snapshot** is a versioned, self-contained bundle of a `ModelPipeline` checkpoint together with its full configuration. It is the artifact that moves between training, evaluation, explanation, and the hub.

## What a snapshot contains

```python
@dataclass
class SnapshotArtifact:
    weights: bytes    # torch.save(state_dict) → bytes
    metadata: bytes   # ModelPipelineConfig.to_dict() → JSON bytes
```

The `weights` are the raw `state_dict` serialized to bytes. The `metadata` is the complete `ModelPipelineConfig` in Hydra-instantiable JSON format — enough to reconstruct the model architecture and instantiate the pipeline without any other context.

## Creating a snapshot

During evaluation, when `save_snapshot=True` in `EvaluationTaskConfig`, the pipeline creates a snapshot from its current state:

```python
artifact = pipeline.ops.to_snapshot()
# artifact.weights  → bytes of the best checkpoint
# artifact.metadata → JSON bytes of the full config
```

Snapshots can also be created manually:

```python
artifact = pipeline.ops.to_snapshot(checkpoint_path="/runs/exp1/best.pt")
```

## Publishing to the hub

`ModelPipelineOps.push_to_hub(artifact, hub_model_name)` uploads the snapshot to Atria Hub:

```python
pipeline.ops.push_to_hub(artifact, hub_model_name="my-org/resnet50-cifar10")
```

On the hub, the snapshot is stored as a versioned model artifact with:
- The weights file
- The metadata JSON (the config)
- Provenance metadata (dataset, training config hash, evaluation metrics)

## Pulling from the hub

```python
from atria_models.api import load_model_pipeline_config
config = load_model_pipeline_config("my-org/resnet50-cifar10", from_hub=True)
pipeline = config.build(labels=dataset.labels)
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
