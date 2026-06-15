---
title: atria_types
---

# atria_types

`atria_types` defines the typed data contracts that flow between every module in Atria — from raw input files through storage, training, and inference. Every sample in the framework is a strongly-typed Pydantic model, not a plain dict or tuple.

## The problem it solves

In typical ML pipelines, data representation is implicit:

```python
# What is "sample"? A dict? What keys? What types?
for sample in dataloader:
    image = sample["image"]       # could be PIL, numpy, tensor?
    label = sample["label"]       # int? list? one-hot?
    annotation = sample["boxes"]  # list? array? what coordinates?
```

Schema drift — where training uses one format and a downstream consumer expects another — is a common source of silent bugs. `atria_types` eliminates this by making the schema explicit and enforced.

## Key abstractions

| Class | Role |
|---|---|
| `BaseDataModel` | Root Pydantic model with `to_row()` / `from_row()` and PyArrow schema support |
| `BaseDataInstance` | Adds `sample_id`, `index`, and typed `annotations` list |
| `ImageInstance` | A sample containing an `Image` and optional annotations |
| `DocumentInstance` | A sample containing a `PDF`, optional `OCR`, and annotations |
| Generic types | `Image`, `PDF`, `Label`, `OCR`, `QAPair`, `BoundingBox` — field-level types |
| Annotation types | `ClassificationAnnotation`, `ObjectDetectionAnnotation`, `EntityLabelingAnnotation`, `QuestionAnsweringAnnotation`, `LayoutAnalysisAnnotation` |

## Relation to other modules

- `atria_datasets` uses `BaseDataInstance` subclasses as the schema for storage rows.
- `atria_models` receives `BaseDataInstance` through the data pipeline and returns typed `ModelOutput`.
- `atria_insights` wraps instances for attribution computation.
- `atria_types` has no dependencies on other Atria packages — it is the lowest layer.
