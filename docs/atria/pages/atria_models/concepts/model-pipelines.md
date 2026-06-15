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

| Pipeline | Registered as | Task |
|---|---|---|
| `ImageClassificationPipeline` | `"image_classification"` | Image classification |
| `SequenceClassificationPipeline` | `"sequence_classification"` | Sequence / text classification |
| `TokenClassificationPipeline` | `"token_classification"` | NER / token labeling |
| `LayoutTokenClassificationPipeline` | `"layout_token_classification"` | Document token labeling with layout features (LiLT, LayoutLMv3) |
| `QuestionAnsweringPipeline` | `"question_answering"` | Extractive QA |

The class hierarchy groups these under two abstract bases: `ImageModelPipeline` (for image tasks) and `SequenceModelPipeline` (for all text/document tasks).

Task-specific pipelines override:
- `_model_build_kwargs()` — extra kwargs for the builder (e.g. `num_labels`)
- `training_step()` / `evaluation_step()` / `predict_step()` — task-specific logic returning `ModelOutput`
- Batching and preprocessing — handled by the model pipeline's transform chain

## ModelPipelineConfig

The config carried by a pipeline specifies:
- `model.builder_type` — which zoo to use
- `model.model_name_or_path` — name or local path
- `model.model_kwargs` — zoo-specific kwargs (e.g. `pretrained=True`)
- `model.frozen_layers` — layers to freeze
- Task-specific fields (number of labels, loss function, etc.)

## Registered pipelines

Pipelines register with flat names in `MODEL_PIPELINES`:

```
image_classification
sequence_classification
token_classification
layout_token_classification
question_answering
```

The pipeline name is the key used in training and evaluation configs to select a pipeline without importing the class directly.
