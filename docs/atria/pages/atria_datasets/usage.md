---
title: Usage
---

# Usage

## Load a registered dataset

```python
from atria_datasets.api import load_dataset

dataset = load_dataset(
    "cifar10/standard",
    data_dir="/data/cache",
    cached_storage_type="MSGPACK",
)

# Iterate the training split
for instance in dataset.train:
    image = instance.image         # Image field
    label = instance.get_annotation_by_type(
        AnnotationType.classification
    ).label
```

## Define a custom dataset

```python
from atria_datasets.core.dataset import ImageDataset
from atria_datasets.core.dataset._common import DatasetConfig
from atria_datasets.registry import DATASETS
from atria_types import ImageInstance

@DATASETS.register("my_dataset/v1")
class MyDataset(ImageDataset):
    class Config(DatasetConfig):
        image_size: int = 224
    __config__ = Config

    def _load_instances(self) -> list[ImageInstance]:
        # return a list of ImageInstance objects
        ...
```

## Use in a training config

```python
from atria_ml.configs import TrainingTaskConfig, DataConfig
from atria_datasets.api import load_dataset_config

config = TrainingTaskConfig(
    data=DataConfig(
        dataset_config=load_dataset_config("my_dataset/v1"),
        train_batch_size=32,
        eval_batch_size=64,
    ),
    model_pipeline=...,
)
```
