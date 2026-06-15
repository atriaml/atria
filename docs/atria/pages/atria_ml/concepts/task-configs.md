---
title: Task Config Hierarchy
---

# Task Config Hierarchy

## TaskConfigBase

The shared root for every task in Atria:

```python
class TaskConfigBase(BaseModel):
    env: RuntimeEnvConfig       # run_dir, seed, device
    data: DataConfig            # dataset + batch sizes + transforms
    logging: LoggingConfig      # log level, WandB / TensorBoard config
    test_run: bool              # if True, use only a few batches
    with_amp: bool              # automatic mixed precision
```

Every downstream config (`TrainingTaskConfig`, `EvaluationTaskConfig`, `ExplanationTaskConfig`) inherits from `TaskConfigBase`. They all have the same dataset and environment — only the task-specific fields differ.

## TrainingTaskConfig

```python
class TrainingTaskConfig(TaskConfigBase):
    model_pipeline: ModelPipelineConfig   # which model to train
    trainer: TrainerConfig                # optimizer, scheduler, LR
    do_train: bool                        # run training loop
    do_test: bool                         # run test evaluation after training
    do_validation: bool                   # run validation during training
    reevaluate_metrics: bool              # recompute metrics even if cached
```

`TrainerConfig` holds all optimization parameters:
- `optimizer` — `ModuleConfig` from the `OPTIMIZERS` registry (Adam, AdamW, SGD, LARS)
- `scheduler` — `ModuleConfig` from the `SCHEDULERS` registry (cosine decay, polynomial decay)
- `learning_rate`, `weight_decay`, `gradient_clip_val`
- `max_epochs`, `eval_every_n_epochs`
- `use_ema`, `ema_decay`

## EvaluationTaskConfig

```python
class EvaluationTaskConfig(TaskConfigBase):
    model_pipeline: ModelPipelineConfig
    eval_checkpoint: str     # path to checkpoint to load before evaluation
    save_snapshot: bool      # if True, save a hub snapshot after evaluation
```

`EvaluationTaskConfig.from_training_config(training_config, eval_checkpoint)` is a convenience factory: copies all shared fields from a `TrainingTaskConfig` and swaps in the checkpoint path. This ensures training and evaluation always use the exact same dataset and model pipeline config.

## RuntimeEnvConfig

```python
class RuntimeEnvConfig(BaseModel):
    run_dir: str      # where to save checkpoints, metrics, configs
    seed: int         # random seed for reproducibility
    device: str       # "cuda", "cpu", "cuda:0"
    num_gpus: int     # for DDP / multi-GPU
```

The `run_dir` is where everything for a single experiment is written: the task config JSON, checkpoints, metric files, and snapshot artifacts.

## Serialization

`TaskConfigBase.save_to_json()` saves the full config (including nested `DataConfig`, `ModelPipelineConfig`, transforms, optimizer) to `{run_dir}/config.json`. `TaskConfigBase.from_json(path)` reconstructs it via Hydra `instantiate`. This makes experiments fully reproducible from their config file alone.
