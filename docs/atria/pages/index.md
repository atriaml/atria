---
title: Atria
sidebar_title: Home
show_title: true
order: 0
---

Atria is a structured ML framework for reproducible training, evaluation, and model explainability. It is organized as a monorepo of focused Python packages, each responsible for one concern — data types, datasets, models, training, or explainability — that compose together through a shared registry and configuration system.

## Why Atria?

Frameworks like PyTorch Lightning and HuggingFace make it easy to get started, but as projects grow they expose structural gaps:

- **No enforced data schema.** Data flowing between preprocessing, training, and serving is typically untyped — dicts, tuples, or loosely typed tensors. Schema drift silently introduces bugs.
- **Coupled task logic.** In PyTorch Lightning, training, evaluation, and prediction logic live in the same `LightningModule`. Switching from training to explanation requires code changes, not config changes.
- **Model zoo fragmentation.** HuggingFace `AutoModel` covers Transformers well; Timm covers vision; Torchvision covers another slice. Combining them in a single project requires ad-hoc glue code.
- **No first-class hub for all artifacts.** HuggingFace Hub is model-centric. Datasets, evaluation snapshots, and explanation runs don't have a unified versioned home.

Atria addresses these by making four things central:

1. **Typed data instances** — every sample is a strongly-typed Pydantic model with explicit row-level serialization to and from storage.
2. **Registry-driven configurability** — every component (dataset, model, transform, explainer) is a named, configurable module. Composition is done in config files, not code.
3. **Decoupled task configs** — `TrainingTaskConfig`, `EvaluationTaskConfig`, and `ExplanationTaskConfig` share the same dataset and model pipeline config. Changing the task is a config field, not a code rewrite.
4. **Hub integration** — datasets and model snapshots are first-class publishable artifacts with versioning and credential management.

## Module Overview

```
atria_registry        ← backbone: ConfigurableModule + ModuleRegistry
atria_types           ← typed data instances, row-level serialization
atria_datasets        ← dataset lifecycle: config → storage → hub
atria_models          ← model builders, task pipelines, snapshots
atria_transforms      ← registered data transforms
atria_metrics         ← task evaluation metrics
atria_ml              ← training / evaluation orchestration via configs
atria_insights        ← explanation pipelines, explainers, XAI metrics
atria_hub             ← versioned artifact store for datasets and models
atria_cli             ← unified command-line entry point
```

Every module except `atria_registry` and `atria_types` depends on the registry. `atria_ml` and `atria_insights` sit at the top of the dependency chain and orchestrate everything else.
