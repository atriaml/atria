---
title: atria_metrics
---

# atria_metrics

`atria_metrics` provides task-level evaluation metrics for Atria training and evaluation pipelines. Metrics are integrated with [PyTorch Ignite](https://pytorch.org/ignite/) and designed for epoch-level aggregation across distributed training.

## What this module provides

Metrics are organized by task domain:

| Domain | Metrics |
|---|---|
| Classification | Accuracy, F1, precision, recall (macro/micro/weighted) |
| Object Detection | COCO mAP, IoU-based metrics |
| Entity Labeling | Sequence-level F1/precision/recall (via seqeval), layout-aware variants |
| Question Answering | Exact match, F1 (DUE Benchmark format) |

## How metrics integrate with training

Metrics plug into the Ignite `Engine` used by `atria_ml`. During a validation or test epoch:

1. `OutputGatherer` collects model predictions and ground-truth labels batch by batch.
2. At epoch end, each metric's `compute()` is called over all gathered outputs.
3. Results are logged and returned as a flat `dict[str, float]` stored in the run directory.

## Key classes

| Class | Role |
|---|---|
| `EpochDictMetric` | Ignite `Metric` subclass that accumulates outputs and calls a `compute_fn` at epoch end |
| `OutputGatherer` | Collects `(predictions, targets)` across batches, handles distributed gathering |

## Relation to EvaluationTaskConfig

`EvaluationTaskConfig` (in `atria_ml`) specifies which metrics to compute. The metrics are task-specific: an image classification evaluation uses `ClassificationMetrics`, while a QA evaluation uses `QuestionAnsweringMetrics`. Because everything is config-driven, the metric set is part of the experiment record.
