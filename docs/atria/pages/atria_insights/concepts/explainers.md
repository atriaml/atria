---
title: Explainers
---

# Explainers

An **explainer** computes feature attributions: a score per input feature (pixel, token, or segment) indicating how much that feature contributed to a model's output.

## Explainer base class

`Explainer` is a `ConfigurableModule` registered in the `EXPLAINERS` registry:

```python
@EXPLAINERS.register("gradient/integrated_gradients")
class IntegratedGradientsExplainer(Explainer):
    class Config(ModuleConfig):
        n_steps: int = 50
        method: str = "gausslegendre"
    __config__ = Config
```

All explainers implement a `explain(forward_func, inputs, targets, baselines, ...)` interface that returns attribution tensors of the same shape as the inputs.

## Gradient-based methods (TorchXAI wrappers)

Wraps Captum attribution methods via TorchXAI:

| Explainer name | Method | Requires baseline |
|---|---|---|
| `gradient/saliency` | Saliency (gradient magnitude) | No |
| `gradient/input_x_gradient` | Input × Gradient | No |
| `gradient/integrated_gradients` | Integrated Gradients | Yes |
| `gradient/deeplift` | DeepLIFT | Yes |
| `gradient/deeplift_shap` | DeepLIFT SHAP | Yes |
| `gradient/gradient_shap` | GradientSHAP | Yes |
| `gradient/guided_backprop` | Guided Backpropagation | No |

## Perturbation-based methods

| Explainer name | Method | Notes |
|---|---|---|
| `perturbation/feature_ablation` | Feature Ablation | Systematic occlusion |
| `perturbation/occlusion` | Occlusion (sliding window) | — |
| `perturbation/lime` | LIME | Surrogate model |
| `perturbation/kernel_shap` | Kernel SHAP | SHAP with LIME framework |

Perturbation methods do not require gradients — they work by masking features and observing model output changes. This makes them applicable to non-differentiable models and black-box APIs.

## Attention-based methods

| Explainer name | Method |
|---|---|
| `attention/raw` | Raw attention weights from the last attention layer |
| `attention/rollout` | Attention rollout across all transformer layers |
| `attention/gradient_weighted` | Gradient-weighted attention |

Attention explainers are specific to transformer architectures. They do not require gradient computation through the model's input embeddings.

## Registry and config

Explainers are selected by name in `ExplanationTaskConfig`:

```yaml
explainer:
  _target_: atria_insights.explainers.IntegratedGradientsConfig
  n_steps: 50
```

Swapping the explainer is a one-line config change — the pipeline, dataset, and metric computation remain identical. This makes systematic comparison across explainer methods straightforward.
