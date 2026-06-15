---
title: Row-Level Conversion
---

# Row-Level Conversion

The core insight behind `atria_types` is that every data instance can be **losslessly flattened to a plain dict of scalars and reconstructed from it**. This is what makes storage backends interchangeable.

## to_row() and from_row()

```python
# Serialize an instance to a storage row
row = instance.to_row()
# row is a flat dict like:
# {
#   "sample_id": "abc-123",
#   "image__bytes": b"...",
#   "image__path": None,
#   "annotations": '[{"type": "classification", "label": "cat"}]',
# }

# Reconstruct from storage
instance2 = ImageInstance.from_row(row)
assert instance == instance2
```

`to_row()` flattens nested Pydantic models using `__` as a separator (e.g. `image.bytes` → `"image__bytes"`). Complex types like `Image` serialize to bytes/path pairs; `annotations` serialize to a JSON string because heterogeneous lists can't be stored as typed columnar fields.

`from_row()` reverses this: it unflattens the dict back to the nested structure that `ImageInstance` expects, then Pydantic validates and constructs the full object.

## PyArrow schema derivation

`BaseDataModel.pa_schema()` derives a PyArrow schema directly from Pydantic field annotations:

```python
schema = ImageInstance.pa_schema()
# pa.schema([
#   ("sample_id", pa.string()),
#   ("index", pa.int64()),
#   ("image__bytes", pa.binary()),
#   ("image__path", pa.string()),
#   ("annotations", pa.string()),
#   ...
# ])
```

This schema is used by `DeltalakeStorageManager` to create typed Delta Lake tables and by `MsgpackShardWriter` to validate records before writing. The same schema definition drives both storage backends — no duplication.

## Why this matters

In frameworks without a typed instance layer, the storage format is defined separately from the data model. When the model changes, the storage schema must be updated independently — and if they drift, deserialization silently returns wrong values.

In Atria, **the Pydantic model class is the schema definition**. Changing a field type on `ImageInstance` immediately changes the PyArrow schema, and `from_row()` will fail loudly on any stored data that no longer matches — catching drift at read time rather than at inference time.

## Immutability and the `ops` pattern

Instances are frozen (`frozen=True`). Modifications return new objects:

```python
updated = instance.update(index=42)
```

For more complex transformations (resizing images, format conversions), the `ops` property exposes a `StandardOps` service object that operates on the instance without mutating it. This keeps instances pure value objects that can be safely cached, hashed, and passed across process boundaries.
