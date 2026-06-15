---
title: atria_transforms
---

# atria_transforms

`atria_transforms` provides data augmentation and preprocessing as first-class registered components. Every transform is a `PydanticConfigurableModule` — its fields ARE its configuration — and is looked up by name in the `DATA_TRANSFORMS` registry.

## Why registered transforms?

In torchvision, a preprocessing pipeline is defined in Python:

```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

This is not serializable without storing the source code. In Atria, the same pipeline is expressed as a config:

```yaml
preprocess_train_transform:
  _target_: atria_transforms.tfs.StandardImageTransformConfig
  resize_height: 224
  resize_width: 224
  stats: imagenet
```

This config is hashed, stored in `DataConfig`, and reproducible across environments.

## Module structure

```
atria_transforms
├── core/
│   ├── _tfs/         ← DataTransform base class
│   └── _data_types/  ← TensorDataModel — typed tensor containers for transforms
├── tfs/              ← Registered transform implementations
│   ├── _image_transforms.py   ← Image resize, crop, normalize
│   ├── _torchvision.py        ← Torchvision transform wrappers
│   ├── _hf_processor.py       ← HuggingFace processor wrappers
│   └── _document_tokenizer.py ← Document tokenization transforms
└── registry/         ← DATA_TRANSFORMS RegistryGroup
```

## TensorDataModel

Between a `BaseDataInstance` (raw data) and a model forward pass, Atria uses `TensorDataModel` objects — typed containers for batched tensors and their metadata. Each pipeline variant defines its own `TensorDataModel` (e.g. `ImageTensorData`, `DocumentTensorData`) so the types flowing through the data pipeline are explicit.

## Key classes

| Class | Role |
|---|---|
| `DataTransform` | Base `PydanticConfigurableModule` for all transforms — fields are the config |
| `TensorDataModel` | Pydantic model holding batched tensors + metadata |
| `DATA_TRANSFORMS` | Registry group for all transforms |

Transforms are attached to `DataConfig.preprocess_train_transform` and `DataConfig.preprocess_eval_transform` — separate transforms for train and eval to support augmentation during training only.
