---
title: Model Builders
---

# Model Builders

A `ModelBuilder` is responsible for constructing a `torch.nn.Module` from a name (or path) and optional kwargs. It abstracts the differences between model zoos so a `ModelPipeline` never needs to know whether its backbone came from Timm, Torchvision, or HuggingFace.

## ModelBuilder

The base class. `ModelBuilder.from_type(builder_type, **kwargs)` dispatches to the right subclass:

| `builder_type` | Builder | Source |
|---|---|---|
| `timm` | `TimmModelBuilder` | [timm](https://github.com/huggingface/pytorch-image-models) — vision models |
| `torchvision` | `TorchvisionModelBuilder` | torchvision model zoo |
| `transformers` | `TransformersModelBuilder` | HuggingFace Transformers |
| `atria` | `AtriaModelBuilder` | Atria-native architectures |
| `local` | `ModelBuilder` (base) | Local checkpoint, no zoo |

## Building a model

Every builder exposes a `build(model_name_or_path, **kwargs)` method. Inside a `ModelPipeline.__init__`, this is called automatically from the config:

```python
self._model = ModelBuilder.from_type(
    builder_type=self.config.model.builder_type,  # e.g. "timm"
    frozen_layers=self.config.model.frozen_layers,
    pretrained_checkpoint=self.config.model.pretrained_checkpoint,
    model_type=self.config.model.model_type,      # task type, used by some builders
).build(
    model_name_or_path=self.config.model.model_name_or_path,  # e.g. "resnet50"
    **self.config.model.model_kwargs,
)
```

## Shared capabilities

All builders support:

- **`frozen_layers`** — specify layer prefixes to freeze after loading weights (`FrozenLayers.none`, `FrozenLayers.all`, or a list of layer name prefixes).
- **`pretrained_checkpoint`** — path or URL to a checkpoint to load on top of the pretrained weights. Supports local paths and fsspec-compatible URLs.
- **`bn_to_gn`** — replace all `BatchNorm` layers with `GroupNorm` (useful for small-batch or variable-batch training).

## Builder-specific behaviour

### TimmModelBuilder

Calls `timm.create_model(model_name=model_name_or_path, pretrained=True, ...)`. Automatically translates `num_labels` → `num_classes` (timm's convention for the output head size).

### TorchvisionModelBuilder

Calls `torch.hub.load("pytorch/vision:v0.10.0", model_name_or_path, ...)`. When `num_labels` is provided in `model_kwargs`, it explicitly replaces the last `Linear` layer of the model with a new `Linear(in_features, num_labels)`. If `num_labels` is absent, the original head is kept and a warning is logged.

### TransformersModelBuilder

Requires `model_type` to specify the **task**, not the architecture. Supported values: `"sequence_classification"`, `"token_classification"`, `"question_answering"`, `"image_classification"`. The task type determines which `AutoModel*` class to use:

| `model_type` | HuggingFace class |
|---|---|
| `sequence_classification` | `AutoModelForSequenceClassification` |
| `token_classification` | `AutoModelForTokenClassification` |
| `question_answering` | `AutoModelForQuestionAnswering` |
| `image_classification` | `AutoModelForImageClassification` |

The architecture is selected by `model_name_or_path` (e.g. `"bert-base-uncased"`, `"microsoft/layoutlm-base-uncased"`). For `image_classification`, the classifier head is also replaced with `Linear(in_features, num_labels)`.

### AtriaModelBuilder

Loads a model config from Atria's own model registry via `load_model_config(model_name_or_path)`. Used for Atria-native transformer encoder architectures with pluggable task heads (`SequenceClassificationHead`, `TokenClassificationHead`, `QuestionAnsweringHead`). Supports `"sequence_classification"`, `"token_classification"`, and `"question_answering"` as `model_type` values (not image classification).

## Contrast with direct model creation

In a typical PyTorch project:

```python
model = timm.create_model("resnet50", pretrained=True)
# Then manually freeze, load extra checkpoint, convert BN, attach head...
```

In Atria, all of this is a config field. Switching from ResNet-50 to ViT-B/16 requires changing `model_name_or_path: "vit_base_patch16_224"` in the config — the builder handles the rest.
