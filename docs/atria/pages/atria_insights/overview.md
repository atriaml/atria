---
title: atria_insights
---

# atria_insights

`atria_insights` extends Atria's config-driven pattern to model explainability. It provides explanation pipelines, explainer methods, and explainability metrics — all as registered, configurable components that slot into the same `TaskConfig` hierarchy as training and evaluation.

## The central idea

Explainability in Atria is not a post-hoc add-on. It is a first-class workflow:

```
ExplanationTaskConfig
    ├── data: DataConfig           ← same dataset as training
    ├── model_pipeline: Config     ← same model as training
    └── explainer: ExplainerConfig ← which attribution method to use
```

`ModelExplainer` orchestrates the explanation run: it loads the dataset and model pipeline via the shared configs, wraps the pipeline in an `ExplanationPipeline`, runs the explainer over the data, computes explainability metrics, and stores results to disk (HDF5).

## Module structure

```
atria_insights
├── configs/
│   └── explanation_task_config.py  ← ExplanationTaskConfig
├── explanation_pipelines/          ← BaseExplanationPipeline + modality variants
├── explainers/
│   ├── _base.py                    ← Explainer base class
│   ├── _torchxai.py                ← TorchXAI / Captum wrappers
│   └── _attn/                      ← Attention-based explainers
├── explainability_metrics/
│   └── _torchxai/                  ← Faithfulness, complexity, robustness, localization
├── baseline_generators/            ← Baseline inputs for gradient-based methods
├── feature_segmentors/             ← Feature grouping (superpixels, token spans)
├── perturbation_robustness/        ← Perturbation-based robustness analysis
├── engines/                        ← Ignite engines for explanation and feature generation
├── storage/                        ← HDF5 caching for attributions and metric results
└── model_explainer.py              ← Top-level orchestrator
```

## Key classes

| Class | Role |
|---|---|
| `ExplanationTaskConfig` | Config root for explanation runs |
| `BaseExplanationPipeline` | Wraps `ModelPipeline` with explanation-specific processing |
| `Explainer` | Base class for attribution methods |
| `ModelExplainer` | Orchestrator: pipeline + explainer + metrics + storage |
| `H5ExplanationStateCacher` | Stores attribution maps to HDF5 |
| `H5MetricDataCacher` | Stores metric values to HDF5 for cross-explainer comparison |

## Relation to torchxai

`atria_insights` uses [TorchXAI](https://github.com/atriaml/torchxai) (the companion library) for the underlying attribution implementations and explainability metric computations. TorchXAI provides efficient Captum-based explainers and batch-scalable XAI metrics. Atria wraps these in its registry/config pattern, adding the storage, pipeline, and task config layers on top.
