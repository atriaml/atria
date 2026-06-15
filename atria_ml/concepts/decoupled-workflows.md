---
title: Decoupled Workflows
---

# Decoupled Workflows

## The problem with coupled task logic

In PyTorch Lightning, training and evaluation behavior are defined in the same `LightningModule`:

```python
class MyModel(LightningModule):
    def training_step(self, batch, batch_idx): ...
    def validation_step(self, batch, batch_idx): ...
    def test_step(self, batch, batch_idx): ...
    def configure_optimizers(self): ...
```

This works well for simple cases, but creates friction when you want to:
- Run evaluation-only without touching training code
- Add explanation on top of an existing model
- Share the same dataset and model config between training and an ablation study

In these scenarios, you end up either subclassing, passing flags, or duplicating code.

## Atria's approach: config swap = workflow swap

In Atria, workflow selection is entirely a config decision:

```python
# Training run
TrainingTaskConfig(
    data=data_config,
    model_pipeline=pipeline_config,
    trainer=TrainerConfig(max_epochs=50),
)

# Evaluation run — same data and model, different task object
EvaluationTaskConfig(
    data=data_config,          # identical
    model_pipeline=pipeline_config,  # identical
    eval_checkpoint="./runs/best.pt",
)

# Explanation run — same data and model again
ExplanationTaskConfig(         # from atria_insights
    data=data_config,          # identical
    model_pipeline=pipeline_config,  # identical
    explainer=explainer_config,
)
```

The `ModelPipeline` and `DataConfig` are plain Pydantic objects — they can be constructed once and passed to any task config. No subclassing, no flags.

## How the engine layer works

`Trainer` and `Evaluator` in `atria_ml/task_pipelines/` receive a `TaskConfig` and:
1. Call `config.build_dataset()` → `Dataset`
2. Call `config.model_pipeline.build(labels=dataset.labels)` → `ModelPipeline`
3. Set up Ignite engines (train, validation, test) with the appropriate handlers
4. Run the loop

The engines are Ignite `Engine` subclasses — pure event-driven computation graphs. Training handlers (EMA, gradient clipping, checkpoint saving, metric logging) are attached as Ignite event handlers, keeping the engine core clean.

## What this enables

- **Reproducibility**: any run can be reproduced from its `config.json`.
- **Ablations**: create N `TrainingTaskConfig` variants differing only in one field, run them all with the CLI.
- **Handoff from training to explanation**: an `EvaluationTaskConfig` can be constructed directly from a `TrainingTaskConfig` via `from_training_config()`, ensuring the same dataset and model are used.
- **Zero-code task switching**: the CLI's `--config` flag selects the task by loading the right config class — no Python changes needed.
