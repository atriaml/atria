---
title: Usage
---

# Usage

## Configure an explanation run

```python
from atria_insights.configs.explanation_task_config import ExplanationTaskConfig
from atria_ml.configs import DataConfig, RuntimeEnvConfig
from atria_datasets.api import load_dataset_config
from atria_models.api import load_model_pipeline_config

config = ExplanationTaskConfig(
    env=RuntimeEnvConfig(run_dir="./runs/exp1/explanations"),
    data=DataConfig(
        dataset_config=load_dataset_config("cifar10/standard"),
        eval_batch_size=16,
    ),
    model_pipeline=load_model_pipeline_config("image/classification/timm"),
    eval_checkpoint="./runs/exp1/checkpoints/best.pt",
    explanation_pipeline="image/classification",
    explainer={
        "_target_": "atria_insights.explainers.IntegratedGradientsConfig",
        "n_steps": 50,
    },
    explainability_metrics=[
        {"_target_": "atria_insights.explainability_metrics.AopcConfig"},
        {"_target_": "atria_insights.explainability_metrics.ComplexityEntropyConfig"},
    ],
)
```

## Run the explainer

```python
from atria_insights.model_explainer import ModelExplainer

explainer = ModelExplainer(config=config)
explainer.run()
# Attributions stored to: ./runs/exp1/explanations/attributions.h5
# Metrics stored to:      ./runs/exp1/explanations/metrics.h5
```

## Run via the CLI

```bash
atria explain --config explanation_config.yaml
```

## Load stored attributions

```python
import h5py

with h5py.File("./runs/exp1/explanations/attributions.h5", "r") as f:
    for sample_id in f.keys():
        attribution = f[sample_id]["attribution"][:]
        # attribution shape: (C, H, W) for image attribution
```

## Compare two explainers

Run two explanation configs differing only in `explainer:`, both pointing to the same checkpoint and dataset. Load both metric HDF5 files and compare per-sample metric values:

```python
import h5py, numpy as np

with h5py.File("runs/ig/metrics.h5") as f_ig, \
     h5py.File("runs/deeplift/metrics.h5") as f_dl:
    aopc_ig = np.array([f_ig[sid]["aopc"][()] for sid in f_ig])
    aopc_dl = np.array([f_dl[sid]["aopc"][()] for sid in f_dl])

print(f"IG AOPC: {aopc_ig.mean():.4f}  DeepLIFT AOPC: {aopc_dl.mean():.4f}")
```
