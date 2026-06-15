---
title: Explainers
---

# Explainers

An **explainer** computes feature attributions: a score per input feature (pixel, token, or segment) indicating how much that feature contributed to a model's output.

## Explainer base class

All explainer configs inherit from `ExplainerConfig`, which itself inherits from `ModuleConfig`. The registered class is the config — there is no separate nested Config class:

```python
@EXPLAINERS.register("grad/integrated_gradients")
class IntegratedGradientsExplainerConfig(GradExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.IntegratedGradientsExplainer"
    type: Literal["grad/integrated_gradients"] = "grad/integrated_gradients"
    n_steps: int = 50
```

All explainer configs implement a `build(model, ...)` method that instantiates the underlying TorchXAI/Captum explainer, returning an object with an `explain(forward_func, inputs, targets, baselines, ...)` interface.

## Gradient-based methods (TorchXAI wrappers)

Wraps Captum attribution methods via TorchXAI:

| Explainer name | Method | Requires baseline |
|---|---|---|
| `grad/saliency` | Saliency (gradient magnitude) | No |
| `grad/input_x_gradient` | Input × Gradient | No |
| `grad/integrated_gradients` | Integrated Gradients | Yes |
| `grad/deeplift` | DeepLIFT | Yes |
| `grad/deeplift_shap` | DeepLIFT SHAP | Yes |
| `grad/gradient_shap` | GradientSHAP | Yes |
| `grad/guided_backprop` | Guided Backpropagation | No |

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
| `attn/raw_attention` | Raw attention weights from the last attention layer |
| `attn/attention_rollout` | Attention rollout across all transformer layers |
| `attn/attention_flow` | Attention flow (graph-based propagation) |

Attention explainers are specific to transformer architectures and require a `TransformersEncoderModel` backbone. They do not require gradient computation through the model's input embeddings.

## Baseline

| Explainer name | Method |
|---|---|
| `random` | Random attribution baseline (uniform noise) |

## Registry and config

Explainers are selected by `type` in the explanation pipeline config:

```yaml
explainer:
  type: "grad/integrated_gradients"
  n_steps: 50
```

Swapping the explainer is a one-line config change — the pipeline, dataset, and metric computation remain identical. This makes systematic comparison across explainer methods straightforward.
