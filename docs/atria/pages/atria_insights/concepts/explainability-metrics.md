---
title: Explainability Metrics
---

# Explainability Metrics

Explainability metrics quantify *how good* an attribution is — independently of human judgment. Atria provides five categories of metrics, built on top of TorchXAI's batch-scalable implementations.

## Why XAI metrics matter

Qualitative inspection of saliency maps is subjective and does not scale. Metrics allow:
- **Systematic comparison** of explainer methods on the same model and dataset.
- **Tracking** attribution quality as models are retrained or fine-tuned.
- **Filtering** explanations that fail basic consistency checks before downstream use.

## Metric categories

### Faithfulness

Faithfulness metrics measure whether the attribution reflects the model's actual decision process:

| Registered name | Question answered |
|---|---|
| `faithfulness/aopc` | AOPC (desc − rand) and ABPC (desc − asc) over perturbation curve |
| `faithfulness/faithfulness_correlation` | Correlation between attribution scores and output change on perturbation |
| `faithfulness/faithfulness_estimate` | Average output change when features are removed in attribution order |
| `faithfulness/infidelity` | Average squared difference between explanation and model sensitivity |
| `faithfulness/monotonicity` | Do cumulative feature additions monotonically increase confidence? |
| `faithfulness/sensitivity_n` | Correlation of attributions with output change for n-feature perturbations |

Note: `faithfulness/aopc` produces both AOPC and ABPC scores in a single pass; they are not separate registered metrics.

### Complexity

Complexity metrics measure whether the explanation is sparse and interpretable:

| Registered name | What it measures |
|---|---|
| `complexity/complexity_entropy` | Shannon entropy of the normalized attribution distribution |
| `complexity/complexity_s` | L2 norm of attribution, normalized by the number of features (Sundararajan) |
| `complexity/effective_complexity` | Number of features needed to explain fraction k of total attribution |
| `complexity/sparseness` | Gini coefficient of attribution magnitudes |

Low complexity = fewer, more concentrated attributions = more interpretable.

### Robustness

| Registered name | What it measures |
|---|---|
| `robustness/sensitivity_max_and_avg` | Maximum and average change in attribution per unit change in input |

This single metric produces both `sensitivity_max` and `sensitivity_avg` values per sample. High sensitivity = explanations change wildly with tiny input changes = unreliable.

### Axiomatic

Axiomatic metrics check whether attributions satisfy theoretical properties:

| Registered name | What it measures |
|---|---|
| `axiomatic/completeness` | Whether attributions sum to the model output difference (completeness axiom) |
| `axiomatic/input_invariance` | Whether attributions are invariant to constant input shifts |
| `axiomatic/monotonicity_corr_and_non_sens` | Monotonicity correlation and non-sensitivity ratio jointly |

### Localization

Localization metrics measure alignment with ground-truth regions (requires segmentation annotations):

| Registered name | What it measures |
|---|---|
| `localization/attr_localization` | Fraction of attribution mass inside the annotated ground-truth region |

This is only applicable to tasks where spatial ground truth is available (object detection, layout analysis).

## Storage and comparison

All metric values are stored per sample and per explainer run via `H5MetricDataCacher` in HDF5 format. This enables:
- Per-sample metric inspection
- Aggregation over the dataset (mean, std, distribution)
- Cross-explainer comparison: run two explainers on the same dataset, load both metric files, compare

## Integration with ExplanationPipelineConfig

Metrics live inside `ExplanationPipelineConfig.explainability_metrics`, which is an `ExplainabilityMetrics` Pydantic model with every metric pre-declared as a named field. All metrics default to `enabled: False`. To activate a metric, set `enabled: True` and override any config fields:

```yaml
explanation_pipeline:
  explainability_metrics:
    aopc:
      enabled: true
      total_feature_bins: 100
      n_random_perms: 10
    complexity_entropy:
      enabled: true
    sensitivity_max_avg:
      enabled: true
      n_perturb_samples: 10
```

The set of available metrics is fixed by the `ExplainabilityMetrics` model. You cannot add new metric types via YAML — you must register them and add a field to `ExplainabilityMetrics`.
