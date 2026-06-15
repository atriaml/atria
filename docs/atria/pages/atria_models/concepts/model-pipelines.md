---
title: Model Pipelines
---

# Model Pipelines

A `ModelPipeline` is the unit that Atria trains, evaluates, and explains. It combines a `ModelBuilder` (which backbone) with task-specific pre/post-processing (how inputs are prepared and outputs interpreted).

## ModelPipeline base

`ModelPipeline` is a `ConfigurableModule` generic over its config type:

```python
class ModelPipeline(ConfigurableModule[T_ModelPipelineConfig]):
    __abstract__ = True
```

**Construction:** Given a `ModelPipelineConfig`, the pipeline:
1. Instantiates the correct `ModelBuilder` from `config.model.builder_type`.
2. Calls `builder.build(model_name_or_path=..., **model_kwargs)` to get a `torch.nn.Module`.
3. Introspects the model's `forward` signature to know which arguments to pass at inference time.

**Key properties:**
- `model` — the underlying `torch.nn.Module`
- `labels` — `DatasetLabels` for the task (class names, id mappings)
- `ops` — `ModelPipelineOps` service object for inference, hub push/pull, and snapshot creation

## Concrete pipeline types

| Pipeline | Task |
|---|---|
| `ImageModelPipeline` | Image classification / detection / segmentation |
| `SequenceModelPipeline` | Sequence classification |
| `QuestionAnsweringPipeline` | Extractive QA |
| `TokenClassificationPipeline` | Named entity recognition, token labeling |

Task-specific pipelines override:
- `_model_build_kwargs()` — extra kwargs for the builder (e.g. `num_labels`)
- `forward()` — wraps the model call, returning a typed `ModelOutput`
- `collate_fn()` — batches `BaseDataInstance` objects into tensors

## ModelPipelineConfig

The config carried by a pipeline specifies:
- `model.builder_type` — which zoo to use
- `model.model_name_or_path` — name or local path
- `model.model_kwargs` — zoo-specific kwargs (e.g. `pretrained=True`)
- `model.frozen_layers` — layers to freeze
- Task-specific fields (number of labels, loss function, etc.)

## Registered pipelines

Pipelines register with named paths in `MODEL_PIPELINES`, typically organized as `{modality}/{task}/{builder}`:

```
image/classification/timm
image/classification/torchvision
sequence/classification/transformers
sequence/token_classification/transformers
sequence/question_answering/transformers
```

This naming convention allows the CLI and training configs to select a pipeline by name without importing the class directly.
