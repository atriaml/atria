---
title: Usage
---

# Usage

## Configure an explanation run

The recommended pattern is the `from_training_task_config` factory — it copies the dataset and model pipeline config from a completed training run and adds the explanation-specific settings:

```python
from atria_insights.configs.explanation_task_config import ExplanationTaskConfig
from atria_insights.explainers._torchxai import IntegratedGradientsExplainerConfig
from atria_insights.explanation_pipelines._common import ExplainabilityMetrics
from atria_insights.explainability_metrics._torchxai._faithfulness import AOPCConfig
from atria_insights.explainability_metrics._torchxai._complexity import ComplexityEntropyConfig

config = ExplanationTaskConfig.from_training_task_config(
    explanation_pipeline_name="image_classification",
    training_task_config=training_config,       # from a completed training run
    exp_name="img_cls_ig_00",
    output_dir="./outputs",
    explainer=IntegratedGradientsExplainerConfig(n_steps=50),
    explainability_metrics=ExplainabilityMetrics(
        aopc=AOPCConfig(enabled=True, total_feature_bins=100),
        complexity_entropy=ComplexityEntropyConfig(enabled=True),
    ),
)
```

For text/document tasks use `"sequence_classification"`, `"token_classification"`, `"layout_token_classification"`, or their `_attn` variants as the pipeline name.

## Run the explainer

```python
from atria_insights.model_explainer import ModelExplainer

explainer = ModelExplainer(
    config=config,
    checkpoint_path="./outputs/exp1/checkpoints/best.pt",
)
explainer.run()
# Attributions stored to: {run_dir}/attributions.h5
# Metrics stored to:      {run_dir}/metrics.h5
```

## Run via the CLI

```bash
atria explain --config explanation_config.yaml
```

## Load stored attributions

```python
import h5py

with h5py.File("./outputs/explanations/attributions.h5", "r") as f:
    for sample_id in f.keys():
        attribution = f[sample_id]["attribution"][:]
        # attribution shape: (C, H, W) for image attribution
```

## Compare two explainers

Run two explanation configs differing only in `explainer`, both pointing to the same checkpoint and dataset. Load both metric HDF5 files and compare per-sample metric values:

```python
import h5py, numpy as np

with h5py.File("runs/ig/metrics.h5") as f_ig, \
     h5py.File("runs/deeplift/metrics.h5") as f_dl:
    aopc_ig = np.array([f_ig[sid]["aopc"][()] for sid in f_ig])
    aopc_dl = np.array([f_dl[sid]["aopc"][()] for sid in f_dl])

print(f"IG AOPC: {aopc_ig.mean():.4f}  DeepLIFT AOPC: {aopc_dl.mean():.4f}")
```
