---
title: Explainability Metrics
---

# Explainability Metrics

Explainability metrics quantify *how good* an attribution is — independently of human judgment. Atria provides four categories of metrics, built on top of TorchXAI's batch-scalable implementations.

## Why XAI metrics matter

Qualitative inspection of saliency maps is subjective and does not scale. Metrics allow:
- **Systematic comparison** of explainer methods on the same model and dataset.
- **Tracking** attribution quality as models are retrained or fine-tuned.
- **Filtering** explanations that fail basic consistency checks before downstream use.

## Metric categories

### Faithfulness

Faithfulness metrics measure whether the attribution reflects the model's actual decision process:

| Metric | Question answered |
|---|---|
| AOPC (Area Over Perturbation Curve) | Does removing high-attribution features decrease output confidence? |
| ABPC (Area Below Perturbation Curve) | Does removing low-attribution features preserve confidence? |
| Faithfulness Correlation | Correlation between attribution scores and output change on perturbation |
| Infidelity | Average squared difference between explanation and model sensitivity |
| Monotonicity | Do cumulative feature additions monotonically increase confidence? |

### Complexity

Complexity metrics measure whether the explanation is sparse and interpretable:

| Metric | What it measures |
|---|---|
| Entropy-based Complexity | Shannon entropy of the normalized attribution distribution |
| Sundararajan Complexity | L2 norm of attribution, normalized by the number of features |
| Effective Complexity | Number of features needed to explain fraction k of total attribution |
| Sparseness | Gini coefficient of attribution magnitudes |

Low complexity = fewer, more concentrated attributions = more interpretable.

### Robustness

Robustness metrics measure stability under small input perturbations:

| Metric | What it measures |
|---|---|
| Sensitivity Max | Maximum change in attribution per unit change in input |
| Sensitivity Average | Average change in attribution per unit change in input |

High sensitivity = explanations change wildly with tiny input changes = unreliable. Robust explainers are important when explanations are used for decision support.

### Localization

Localization metrics measure alignment with ground-truth regions (requires segmentation annotations):

| Metric | What it measures |
|---|---|
| Attribution Localization | Fraction of attribution mass inside the annotated ground-truth region |
| Attribution Locality | Proportion of top-k attributed features within the target region |

These are only applicable to tasks where spatial ground truth is available (object detection, layout analysis).

## Storage and comparison

All metric values are stored per sample and per explainer run via `H5MetricDataCacher` in HDF5 format. This enables:
- Per-sample metric inspection
- Aggregation over the dataset (mean, std, distribution)
- Cross-explainer comparison: run two explainers on the same dataset, load both metric files, compare

## Integration with ExplanationTaskConfig

Metrics to compute are specified in `ExplanationTaskConfig`:

```yaml
explainability_metrics:
  - _target_: atria_insights.explainability_metrics.AopcConfig
    n_perturbation_steps: 10
  - _target_: atria_insights.explainability_metrics.ComplexityEntropyConfig
  - _target_: atria_insights.explainability_metrics.SensitivityMaxConfig
    n_perturb_samples: 10
```

Like explainers, metrics are registered `ConfigurableModule` instances — adding or removing a metric is a config list change.
