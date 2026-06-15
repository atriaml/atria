---
title: Storage Backends
---

# Storage Backends

Atria supports multiple storage backends for caching dataset instances. The backend is selected by `cached_storage_type` in `load_dataset` or `DataConfig`.

## DeltalakeStorageManager

**Format:** Parquet files (Delta Lake) for metadata + WebDataset TAR shards for binary content.

**Best for:** Large datasets where you want schema versioning, partial reads, or SQL-style queries on metadata fields, without paying the cost of storing large binary blobs inside the columnar table.

### Hybrid metadata + shard layout

A naive Delta Lake implementation would embed image bytes directly as binary columns in the Parquet files. This is problematic: large blobs bloat Parquet row groups, make columnar reads slow, and force the entire binary into memory even when only metadata is needed.

Atria avoids this with a **hybrid layout**: the Delta Lake table stores only metadata and lightweight pointers; the actual binary content lives in separate WebDataset TAR shards.

```
{config_name}/
├── delta/
│   └── {split}/          ← Delta Lake table (Parquet + transaction log)
│       ├── _delta_log/
│       └── part-*.parquet
├── shards/
│   └── {split}/          ← WebDataset TAR shards (binary content)
│       ├── 000000-000001.tar
│       └── 000000-000002.tar
└── data/
    └── {split}/          ← Direct file copies (PDFs, non-image binaries)
```

### How binary content is routed during write

`DeltalakeWriterWorker` inspects each instance's `to_row()` output for field pairs of the form `{field}__file_path` + `{field}__content`:

- **If `content` is present** (e.g. an image that has been loaded into memory): the bytes are written into a WebDataset TAR shard via `wds.ShardWriter`. The corresponding `file_path` column in the Delta Lake row is replaced with a shard pointer URL encoding the tar file path, byte offset, and length:

  ```
  shards/train/000000-000001.tar?offset=1536&length=49152
  ```

  The `content` column is set to `None` in the Parquet row — the blob never enters Delta Lake.

- **If only `file_path` is present** (e.g. a PDF that is not pre-loaded): the file is copied into `data/{split}/` and the relative path is stored in the row. The Delta Lake table stores just the string path.

### How binary content is retrieved during read

`InMemoryDeltalakeReader` / `LocalDeltalakeReader` load the Parquet rows as a Pandas DataFrame. For any `file_path` column, the stored path is resolved against `{storage_dir}/{config_name}/`:

- A shard pointer (`shards/train/…tar?offset=…&length=…`) is resolved to the full local path of the tar file. The `Image` type then performs a **byte-range read** — seeking directly to `offset` and reading `length` bytes — without scanning the entire shard.
- A plain file path (`data/train/img_001.jpg`) is resolved to the full local file path and read normally.

This means random access into a TAR shard is O(1): no sequential scan, no full shard load. The Delta Lake table remains a fast, queryable metadata store, and the TAR shards serve as a compact binary store with efficient random access.

### Write parallelism

Writing is parallelized with Ray (`RayParallelDeltalakeWriter`). Each Ray actor runs a `DeltalakeWriterWorker` independently, writing its own shard file. Row dicts (with shard pointers) are gathered by a coordinator process and flushed to the Delta Lake table in batches sized to stay within a configurable memory limit.

**Trade-off:** More complex on-disk layout than msgpack; better for structured or mixed datasets where you need both fast metadata queries and efficient binary access.

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
