---
title: Dataset Lifecycle
---

# Dataset Lifecycle

## 1. Config-based creation

Every dataset is a `DatasetConfig` — a `ModuleConfig` subclass registered in the `DATASETS` registry. Loading a dataset starts with a name lookup:

```python
from atria_datasets.api import load_dataset_config, load_dataset

# Returns the registered DatasetConfig for "cifar10/standard"
config = load_dataset_config("cifar10/standard")

# Full lifecycle: download → convert → cache → split
dataset = load_dataset("cifar10/standard", data_dir="/data/cache")
```

`DatasetConfig` holds dataset-level metadata: `dataset_name`, `config_name`, sample limits (`max_train_samples`), random `seed`. It can be extended with dataset-specific fields (extra fields are allowed via `model_config = ConfigDict(extra="allow")`).

## 2. Download

`DownloadManager` handles raw data acquisition:

- **HTTP** — standard URL downloads with progress tracking
- **FTP** — FTP file downloads
- **Google Drive** — Drive file IDs with authentication

Downloaded files are cached locally. Re-running `load_dataset` with the same config reuses the cache unless `overwrite_existing_cached=True`.

## 3. Convert and validate

Raw files are converted to `BaseDataInstance` subclasses (e.g. `ImageInstance`). Pydantic validates each field at construction time. If a field type is wrong or a required field is missing, an error is raised immediately — before anything is written to storage.

## 4. Cache to storage

Validated instances are written to a storage backend via their `to_row()` method. The storage type is selected by `cached_storage_type`:

- `FileStorageType.DELTALAKE` — columnar Parquet-backed Delta Lake table
- `FileStorageType.MSGPACK` — binary shard files, faster for image-heavy datasets

The storage backend uses `BaseDataInstance.pa_schema()` to create a typed schema before writing. Subsequent loads read from cache, skipping download and conversion.

## 5. Split

If the dataset does not include a pre-defined validation split, `StandardSplitter` creates one:

```python
train, validation = StandardSplitter(split_ratio=0.9)(dataset.train)
```

Split information is persisted alongside the cache so subsequent loads reproduce the exact same split.

## 6. SplitIterator

The final `Dataset` object exposes a dict of `SplitIterator` objects, keyed by `DatasetSplitType` (train, validation, test). Each `SplitIterator` is a PyTorch `Dataset`-compatible iterable that reads from the storage backend and applies on-the-fly transforms.

```python
for instance in dataset.train:
    # instance is a fully validated BaseDataInstance subclass
    image = instance.image
    label = instance.get_annotation_by_type(AnnotationType.classification).label
```

## DataConfig — the training-time wrapper

In training contexts, `DataConfig` wraps `DatasetConfig` with additional parameters: batch sizes, number of workers, preprocess transforms, and split ratio. `atria_ml` uses `DataConfig` rather than `DatasetConfig` directly.
