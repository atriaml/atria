---
title: atria_models
---

# atria_models

`atria_models` defines how models are selected, built, and used for inference in Atria. It separates two concerns that are typically coupled in other frameworks:

- **Model builder** — *what* architecture and weights to load (Timm, Torchvision, Transformers, or custom Atria models)
- **Model pipeline** — *how* that architecture is used for a specific task (image classification, sequence classification, QA, token labeling)

This separation means you can swap a ResNet for a ViT by changing a config field, without touching the task-specific logic, and swap a classification head for a QA head by changing the pipeline type.

## Module structure

```
atria_models
├── core/
│   ├── model_builders/     ← ModelBuilder and zoo-specific subclasses
│   ├── model_pipelines/    ← ModelPipeline base + task-specific subclasses
│   └── models/             ← Custom Atria model implementations (Transformer variants)
├── utilities/
│   └── _checkpoints.py     ← Checkpoint save/load (bytes, path, URL)
└── api/
    └── models.py           ← load_model_pipeline, load_model_pipeline_config
```

## Key classes

| Class | Role |
|---|---|
| `ModelBuilder` | Resolves `builder_type` to the right zoo and builds a `torch.nn.Module` |
| `TimmModelBuilder` | Builds from [timm](https://github.com/huggingface/pytorch-image-models) model zoo |
| `TorchvisionModelBuilder` | Builds from torchvision model zoo |
| `TransformersModelBuilder` | Builds from HuggingFace Transformers model zoo |
| `AtriaModelBuilder` | Builds Atria-native model architectures |
| `ModelPipeline` | Base class: config-driven component wrapping a model builder + task logic |
| `ImageModelPipeline` | Pipeline for image-based tasks |
| `SequenceModelPipeline` | Pipeline for text sequence tasks |
| `SnapshotArtifact` | Weights + metadata bundle for hub publishing |

## Registry

Model pipelines register themselves in `MODEL_PIPELINES`:

```python
@MODEL_PIPELINES.register("image/classification/timm")
class ImageClassificationPipeline(ImageModelPipeline):
    __config__ = ImageClassificationConfig
```

Loading:

```python
from atria_models.api import load_model_pipeline_config
config = load_model_pipeline_config("image/classification/timm")
```
