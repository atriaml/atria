---
title: Metric Types
---

# Metric Types

## EpochDictMetric

`EpochDictMetric` is the base building block. It extends Ignite's `Metric` with a pattern suited to Atria's output format:

```
update(batch_output) → accumulate in _fn_inputs
compute()            → compute_fn(*all_inputs) → float
reset()              → clear _fn_inputs
```

The `compute_fn` is any callable that takes the full epoch's predictions and targets and returns a scalar. This design lets you use sklearn metrics, torchmetrics, or custom functions as `compute_fn`.

## OutputGatherer

`OutputGatherer` is a helper that sits between the model engine and the metrics. It:

1. Collects `(logits, labels)` pairs (or any structured output) across all batches in an epoch.
2. At epoch end, concatenates them into full-epoch tensors.
3. In distributed training, gathers tensors across ranks before computing metrics.

This avoids the common bug of computing metrics per-batch and averaging — which gives wrong results for imbalanced batches or metrics that aren't decomposable (like F1 across all samples).

## Metric organization

Each task domain has its own metric module under `atria_metrics/core/`:

```
core/
├── classification/    ← accuracy, F1, precision/recall
├── detection/         ← COCO eval (cocoeval.py)
├── entity_labeling/   ← seqeval, layout precision/recall/F1
└── question_answering/← DUE benchmark QA eval (exact match, F1)
```

Metrics are registered in domain-specific `RegistryGroup` instances so training configs can select them by name rather than importing the class.

## Distributed-safe computation

`EpochDictMetric` uses `ignite.distributed` to gather tensors from all processes before computing. This means the same metric code works in single-GPU, multi-GPU (DDP), and multi-node training without modification — a consequence of Atria delegating distributed coordination to Ignite rather than implementing it in each metric.
