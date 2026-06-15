---
title: Data Instances
---

# Data Instances

A **data instance** is a single sample in Atria — an image, a document, or any other modality — represented as a typed Pydantic object.

## Hierarchy

```
BaseDataModel
└── BaseDataInstance
    ├── ImageInstance
    └── DocumentInstance
```

### BaseDataModel

The root of the type hierarchy. Extends Pydantic's `BaseModel` with:

- **`to_row()`** — serialize the instance to a flat dict of PyArrow-compatible scalars, suitable for writing to Delta Lake or msgpack storage.
- **`from_row(row)`** — reconstruct an instance from a flat storage row, unflattening nested fields automatically.
- **`pa_schema()`** — derive a PyArrow schema from field annotations, used to create typed columnar tables.
- **`ops`** — a bound service object (`StandardOps`) for common transformations without mutating the immutable model.

Model config is strict: `extra="forbid"`, `frozen=True`, `strict=True`. Unknown fields are rejected at validation time, and instances are immutable after creation.

### BaseDataInstance

Adds the fields common to all samples:

- `sample_id` — UUID, auto-generated if not provided.
- `index` — optional integer position in the dataset.
- `annotations` — a list of typed `Annotation` objects, serialized to JSON string when stored (because columnar stores don't support heterogeneous nested lists).

The `annotations` field is the extensibility point: a single instance can carry multiple annotation types simultaneously (e.g. both a classification label and bounding boxes).

### ImageInstance and DocumentInstance

Concrete subclasses that add modality-specific fields:

- `ImageInstance` adds an `Image` field (lazy-loaded from path or bytes).
- `DocumentInstance` adds `PDF` and optionally `OCR` (structured text recognition output).

## Generic field types

These are the building blocks of instance fields:

| Type | Stores |
|---|---|
| `Image` | Image bytes or file path, with lazy decoding |
| `PDF` | PDF bytes or path |
| `Label` | A single classification label (string or int) |
| `OCR` | Structured OCR output with word-level bounding boxes |
| `QAPair` | A question–answer pair |
| `BoundingBox` | A (x1, y1, x2, y2) rectangle |

Each type knows how to serialize and deserialize itself for storage (via `to_row` / `from_row` on the containing instance).

## Annotation types

Annotations are task-specific labels attached to a `BaseDataInstance`. They are polymorphic — a single `annotations` list can hold any mix:

| Annotation | Used for |
|---|---|
| `ClassificationAnnotation` | Image or document classification |
| `ObjectDetectionAnnotation` | Bounding box detection |
| `EntityLabelingAnnotation` | Named entity recognition / token labeling |
| `QuestionAnsweringAnnotation` | Extractive or abstractive QA |
| `LayoutAnalysisAnnotation` | Document layout parsing |

Retrieve an annotation from an instance:

```python
ann = instance.get_annotation_by_type(AnnotationType.classification)
```

This raises `AnnotationNotFoundError` if the annotation is absent, making missing-annotation bugs explicit rather than silent.
