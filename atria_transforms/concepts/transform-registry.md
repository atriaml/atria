---
title: Transform Registry
---

# Transform Registry

## DataTransform

`DataTransform` is the base class for all transforms. It inherits from `PydanticConfigurableModule`, which means the transform class itself IS the config — its Pydantic fields are the hyperparameters, with no separate nested Config class:

```python
@DATA_TRANSFORMS.register("standard_image_transform")
class StandardImageTransform(DataTransform[np.ndarray]):
    to_rgb: bool = True
    do_normalize: bool = True
    do_resize: bool = True
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "imagenet"
    resize_height: int = 224
    resize_width: int = 224

    def __call__(self, input: Any) -> np.ndarray:
        # convert instance → numpy array
        ...
```

A transform takes arbitrary input and returns a typed output (often a `TensorDataModel` or numpy array). The `data_model` property declares the output type.

## DATA_TRANSFORMS registry

The `DATA_TRANSFORMS` `RegistryGroup` holds all registered transforms. Because transforms are `PydanticConfigurableModule` subclasses (not `ConfigurableModule`), they register the class itself as the config:

```python
from atria_transforms.registry import DATA_TRANSFORMS

config = DATA_TRANSFORMS.load_module_config("standard_image_transform")
transform = config.build()
# or equivalently:
transform = StandardImageTransform(resize_height=256)
```

## Train vs. eval transforms

`DataConfig` holds separate transforms for training and evaluation:

- `preprocess_train_transform` — applied during training; can include random augmentations
- `preprocess_eval_transform` — applied during evaluation; deterministic

Both are `DataTransform | None`. Setting them to different transforms is the standard pattern for data augmentation:

```python
DataConfig(
    preprocess_train_transform=StandardImageTransform(do_resize=True, stats="imagenet"),
    preprocess_eval_transform=StandardImageTransform(do_resize=True, stats="imagenet"),
)
```

## Available transform families

| Family | Transforms |
|---|---|
| Image | `StandardImageTransform` — resize, normalize to configurable stats (ImageNet, CLIP, custom) |
| Torchvision wrappers | Any `torchvision.transforms` wrapped as a `DataTransform` |
| HuggingFace processor | `AutoImageProcessor`, `AutoTokenizer` wrappers |
| Document tokenizer | LayoutLM/BERT-style document tokenization for OCR + bounding boxes |

## Serialization

Because every transform is a `PydanticConfigurableModule`, it serializes to a Hydra-instantiable dict:

```python
transform.to_dict()
# {"_target_": "atria_transforms.tfs.StandardImageTransform", "resize_height": 224, ...}
```

This dict is stored in `DataConfig` and therefore in `TaskConfig`, making the full preprocessing pipeline part of the experiment record.
