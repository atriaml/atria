---
title: atria_ml
---

# atria_ml

`atria_ml` is the orchestration layer of the Atria framework. It turns configuration objects into running training, evaluation, and explanation pipelines. The central design principle: **task type is a config field, not a code path**.

## The core insight

In PyTorch Lightning, train / evaluate / predict behaviors live in the same `LightningModule`. Adding explanation requires adding more methods or subclassing. In Atria:

```
TrainingTaskConfig    → Trainer engine
EvaluationTaskConfig  → Evaluator engine
ExplanationTaskConfig → ModelExplainer (in atria_insights)
```

All three share the same `DataConfig` and `ModelPipelineConfig`. Switching from training to evaluation is changing one config object — the model pipeline and dataset are unchanged. Switching to explanation adds the explainer config but reuses everything else.

## Module structure

```
atria_ml
├── configs/
│   ├── _task.py      ← TaskConfigBase, TrainingTaskConfig, EvaluationTaskConfig
│   ├── _data.py      ← DataConfig
│   ├── _trainer.py   ← TrainerConfig (hardware / optimization settings)
│   └── _env.py       ← RuntimeEnvConfig (run directory, seed, device)
├── task_pipelines/
│   ├── _trainer.py   ← Trainer: wraps Ignite engine, runs train + eval loop
│   └── _evaluator.py ← Evaluator: runs test engine + computes metrics
├── training/
│   ├── engines/      ← Ignite engine subclasses for train, val, test, predict
│   └── handlers/     ← EMA, terminate-on-NaN, checkpoint handlers
├── optimizers/       ← OPTIMIZERS registry (Adam, AdamW, SGD, LARS, …)
└── schedulers/       ← SCHEDULERS registry (cosine, polynomial decay, …)
```

## Key components

| Class | Role |
|---|---|
| `TaskConfigBase` | Pydantic base holding `data`, `env`, `logging` |
| `TrainingTaskConfig` | Adds `model_pipeline`, `trainer`, and `do_train/test/validation` flags |
| `EvaluationTaskConfig` | Adds `eval_checkpoint` and `save_snapshot` |
| `TrainerConfig` | Optimizer, scheduler, learning rate, gradient clipping, EMA |
| `RuntimeEnvConfig` | `run_dir`, random seed, precision (`with_amp`) |
| `DataConfig` | Dataset config + batch sizes + transforms |
| `Trainer` | Runs the full train + validation loop using Ignite engines |
| `Evaluator` | Runs test inference + metric computation |
