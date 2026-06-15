---
title: Storage Backends
---

# Storage Backends

Atria supports multiple storage backends for caching dataset instances. The backend is selected by `cached_storage_type` in `load_dataset` or `DataConfig`.

## DeltalakeStorageManager

**Format:** Parquet files managed as a Delta Lake table.

**Best for:** Large datasets where you want schema versioning, partial reads, or SQL-style queries on metadata fields. Columnar format is efficient for field-selective reads (e.g. loading labels without loading image bytes).

**How it works:**
1. Creates a Parquet schema from `BaseDataInstance.pa_schema()`.
2. Writes batches of `to_row()` dicts as Parquet row groups.
3. Delta Lake transaction log enables atomic writes and time-travel reads.
4. `DeltalakeReader` streams rows on read, reconstructing instances via `from_row()`.

**Trade-off:** Higher write overhead than msgpack; better for structured tabular data or metadata-heavy datasets.

## MsgpackShardWriter

**Format:** Binary msgpack shard files (`.mp` extension).

**Best for:** Image and binary-heavy datasets where raw throughput matters. Msgpack encodes arbitrary Python objects compactly with fast serialization.

**How it works:**
1. Writes validated `to_row()` dicts as msgpack records into numbered shard files.
2. Shards are sized automatically based on record count.
3. `MsgpackShardReader` iterates shard files, decoding and reconstructing instances.

**Trade-off:** No schema enforcement at the storage layer — relies on Pydantic validation at read time. Faster for image pipelines.

## WebDataset

**Format:** TAR archives (`.tar` shards) compatible with the WebDataset library.

Available as `FileStorageType.WEBDATASET` for distributed training scenarios where data lives on object storage (S3, GCS) and needs to be streamed efficiently across many workers.

## HuggingFace adapter

`HuggingfaceDataset` wraps any HuggingFace dataset (accessed via `datasets.load_dataset`) without local caching:

```python
@DATASETS.register("imagenet/hf")
class ImageNetHF(HuggingfaceDataset):
    class Config(DatasetConfig):
        hf_dataset_name: str = "imagenet-1k"
    __config__ = Config
```

The adapter converts HuggingFace rows to Atria `BaseDataInstance` objects on the fly. It can then be re-cached to Deltalake or msgpack for subsequent runs.

## Choosing a backend

| Scenario | Recommended backend |
|---|---|
| Structured / tabular data | `DELTALAKE` |
| Image-heavy, high throughput | `MSGPACK` |
| Already stored in HuggingFace Hub | `HuggingfaceDataset` adapter |
| Distributed training over object storage | `WEBDATASET` |
