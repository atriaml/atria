---
title: Usage
---

# Usage

## Define a training config (Python)

```python
from atria_ml.configs import TrainingTaskConfig, DataConfig, TrainerConfig
from atria_ml.configs._env import RuntimeEnvConfig
from atria_datasets.api import load_dataset_config
from atria_models.api import load_model_pipeline_config

config = TrainingTaskConfig(
    env=RuntimeEnvConfig(run_dir="./runs/exp1", seed=42),
    data=DataConfig(
        dataset_config=load_dataset_config("cifar10/standard"),
        train_batch_size=64,
        eval_batch_size=128,
    ),
    model_pipeline=load_model_pipeline_config(
        "image/classification/timm",
        model_name_or_path="resnet50",
    ),
    trainer=TrainerConfig(
        learning_rate=1e-3,
        max_epochs=50,
    ),
)

config.save_to_json("./runs/exp1/config.json")
```

## Run via the Trainer

```python
from atria_ml.task_pipelines import Trainer

trainer = Trainer(config=config)
trainer.run()
```

## Run evaluation from a saved config

```python
from atria_ml.configs import EvaluationTaskConfig

eval_config = EvaluationTaskConfig.from_json("./runs/exp1/config.json")
# or construct from training config:
eval_config = EvaluationTaskConfig.from_training_config(
    config, eval_checkpoint="./runs/exp1/checkpoints/best.pt"
)
```

## YAML config (for CLI use)

```yaml
# train_config.yaml
_target_: atria_ml.configs.TrainingTaskConfig
env:
  _target_: atria_ml.configs.RuntimeEnvConfig
  run_dir: ./runs/exp1
  seed: 42
data:
  _target_: atria_ml.configs.DataConfig
  dataset_config:
    _target_: atria_datasets...Cifar10Config
  train_batch_size: 64
model_pipeline:
  _target_: atria_models...ImageClassificationConfig
  model:
    builder_type: timm
    model_name_or_path: resnet50
trainer:
  _target_: atria_ml.configs.TrainerConfig
  learning_rate: 0.001
  max_epochs: 50
```
