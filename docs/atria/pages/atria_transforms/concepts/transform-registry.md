---
title: Transform Registry
---

# Transform Registry

## DataTransform

`DataTransform` is the base class for all transforms. It is a `ConfigurableModule` and therefore has a paired `ModuleConfig`:

```python
@TRANSFORMS.register("image/resize_normalize")
class ImageResizeNormalize(DataTransform):
    class Config(ModuleConfig):
        size: int = 224
        mean: list[float] = [0.485, 0.456, 0.406]
        std: list[float] = [0.229, 0.224, 0.225]
    __config__ = Config

    def __call__(self, instance: BaseDataInstance) -> TensorDataModel:
        # convert instance → tensor data model
        ...
```

A transform takes a `BaseDataInstance` (or batch thereof) and returns a `TensorDataModel` — the typed tensor bundle ready for model input.

## TRANSFORMS registry

The `TRANSFORMS` `RegistryGroup` holds all registered transforms. Transforms are loaded the same way as datasets and model pipelines:

```python
from atria_transforms.api import load_transform_config

config = load_transform_config("image/resize_normalize", size=256)
transform = config.build()
```

## Train vs. eval transforms

`DataConfig` holds separate transforms for training and evaluation:

- `preprocess_train_transform` — applied during training; can include random augmentations
- `preprocess_eval_transform` — applied during evaluation; deterministic

Both are `DataTransform | None`. Setting them to different transforms is the standard pattern for data augmentation:

```python
DataConfig(
    preprocess_train_transform=load_transform_config("image/augment_and_normalize"),
    preprocess_eval_transform=load_transform_config("image/resize_normalize"),
)
```

## Available transform families

| Family | Transforms |
|---|---|
| Image | Resize, center crop, random crop, normalize, color jitter |
| Torchvision wrappers | Any `torchvision.transforms` wrapped as a `DataTransform` |
| HuggingFace processor | `AutoImageProcessor`, `AutoTokenizer` wrappers |
| Document tokenizer | LayoutLM/BERT-style document tokenization for OCR + bounding boxes |

## Serialization

Because every transform is a `ModuleConfig`, it serializes to a Hydra-instantiable dict:

```python
config.to_dict()
# {"_target_": "atria_transforms.tfs.ImageResizeNormalizeConfig", "size": 224, ...}
```

This dict is stored in `DataConfig` and therefore in `TaskConfig`, making the full preprocessing pipeline part of the experiment record.
