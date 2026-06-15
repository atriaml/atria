---
title: atria_datasets
---

# atria_datasets

`atria_datasets` manages the full lifecycle of a dataset in Atria: downloading raw data, validating and converting it to typed `BaseDataInstance` objects, caching to a persistent storage backend, splitting into train/validation/test, and publishing to the hub.

## The lifecycle at a glance

```
load_dataset_config("cifar10/standard")
       ↓
DatasetConfig  (registered in DATASETS registry)
       ↓ .build()
Dataset.load()
       ├── DownloadManager  → raw files
       ├── convert to BaseDataInstance subclasses
       ├── storage backend cache (Deltalake / msgpack)
       ├── StandardSplitter → train / validation / test
       └── SplitIterator    → PyTorch DataLoader-compatible iterable
```

Every step is parameterized through the `DatasetConfig` and the `DataConfig` that wraps it.

## What this module provides

| Class | Role |
|---|---|
| `DatasetConfig` | `ModuleConfig` subclass describing a dataset; registered in `DATASETS` |
| `Dataset` | Container holding `SplitIterator` objects per split |
| `ImageDataset` / `DocumentDataset` | Typed dataset base classes |
| `HuggingfaceDataset` | Adapter that wraps HuggingFace datasets as Atria datasets |
| `DownloadManager` | Unified downloader (HTTP, FTP, Google Drive) with caching |
| `DeltalakeStorageManager` | Columnar cached storage via Delta Lake / Parquet |
| `MsgpackShardWriter` | Binary shard storage for high-throughput pipelines |
| `StandardSplitter` | Splits a dataset into train/validation given a ratio |

## Contrast with HuggingFace Datasets

HuggingFace `load_dataset("cifar10")` returns a dict of Arrow tables. Atria does the same but couples the result to a typed schema (`BaseDataInstance.pa_schema()`) so that validation errors appear at load time, not at training time. The `HuggingfaceDataset` adapter bridges the two: any HuggingFace dataset can be wrapped and cached as Atria typed instances.
