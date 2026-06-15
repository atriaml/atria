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

## TransformersModelBuilder specifics

Transformers models require a `model_type` (e.g. `"bert"`, `"roberta"`, `"layoutlm"`) to select the right `AutoModel` class. The builder calls `AutoModel.from_pretrained(model_name_or_path)` and applies any Atria-specific head modifications on top.

## Contrast with direct model creation

In a typical PyTorch project:

```python
model = timm.create_model("resnet50", pretrained=True)
# Then manually freeze, load extra checkpoint, convert BN, attach head...
```

In Atria, all of this is a config field. Switching from ResNet-50 to ViT-B/16 requires changing `model_name_or_path: "vit_base_patch16_224"` in the config — the builder handles the rest.
