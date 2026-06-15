---
title: Explanation Pipelines
---

# Explanation Pipelines

An `ExplanationPipeline` wraps a `ModelPipeline` with the additional processing needed to produce and evaluate attributions. It is a `ConfigurableModule`, registered in `EXPLANATION_PIPELINES`.

!!! note "Model requirement"
    Text explanation pipelines (`SequencePipeline`, `AttnSequencePipeline`) require the underlying model to be a [`TransformersEncoderModel`](../../atria_models/concepts/transformer-encoder.md) — Atria's native transformer base that provides the embedding-space forward path and attention extraction interface. HuggingFace `AutoModel*` wrappers do not satisfy this contract.

## BaseExplanationPipeline

```python
class BaseExplanationPipeline(ConfigurableModule[T_ExplanationPipelineConfig]):
    def __init__(self, config, model_pipeline: ModelPipeline): ...
```

The base class holds a reference to the `ModelPipeline` and provides:

- **`forward_func`** — a callable wrapping the model pipeline's forward pass in a form that attribution methods can differentiate.
- **Attribution step** — runs the explainer over a batch and returns `BatchExplanationState` (attributions + metadata).
- **Metric step** — computes explainability metrics on stored attributions.
- **Caching** — attributions and metric results are written to HDF5 via `ExplanationStateCacher` and `MetricDataCacher`.

## Concrete pipeline types

### ImagePipeline (`image_classification`)

Handles image attribution:

- Converts `BaseDataInstance` → image tensors via the model pipeline's transform.
- Manages **baseline generation**: `BaselineGenerator` subclasses produce the reference input (zero, Gaussian noise, blurred image) for gradient-based methods that require a baseline.
- Manages **feature segmentation**: `FeatureSegmentor` subclasses group pixels into superpixels or regions, enabling segment-level attribution rather than pixel-level.
- Returns `BatchExplanationState` with per-pixel (or per-segment) attribution maps.

### SequencePipeline

Gradient-based token-level text attribution. Registered names:

| Name | Task |
|---|---|
| `sequence_classification` | Text / sequence classification |
| `token_classification` | NER / token labeling |
| `layout_token_classification` | Document token labeling (LiLT, LayoutLMv3) |
| `question_answering` | Extractive QA |

- Converts tokenized sequences to embedding-space representations.
- Runs gradient-based methods in embedding space (for `IntegratedGradients`, `DeepLIFT`, etc.).
- Aggregates embedding-dimension attributions to token-level scores.
- Returns `BatchExplanationState` with per-token attribution scores.

### AttnSequencePipeline

Attention-based alternative to gradient computation. Registered names:

| Name | Task |
|---|---|
| `sequence_classification_attn` | Text / sequence classification |
| `token_classification_attn` | NER / token labeling |
| `question_answering_attn` | Extractive QA |

- Extracts attention weights directly from transformer layers (no gradient computation needed).
- Supports attention rollout (multiplying attention matrices across layers) for multi-layer models.
- Returns the same `BatchExplanationState` format as `SequencePipeline`, so downstream metric computation is identical.

## Baseline generators

Gradient-based attribution methods (Integrated Gradients, DeepLIFT, GradientSHAP) require a **baseline** — a reference input representing "no signal". `BaselineGenerator` subclasses in `atria_insights/baseline_generators/` produce baselines appropriate for each modality:

| Generator | Baseline |
|---|---|
| `ZeroBaselineGenerator` | All-zero tensor |
| `GaussianNoiseGenerator` | Gaussian noise sample |
| `BlurBaselineGenerator` | Blurred version of the input |
| `SequenceBaselineGenerator` | PAD token or mask token embedding |

## Feature segmentors

For image attribution, raw pixel-level maps are often noisy. `FeatureSegmentor` subclasses group pixels into semantically meaningful regions before attribution:

- SLIC superpixel segmentation
- Grid-based segments
- Segment-anything (SAM) masks

Segment-level attribution is more robust and easier to visualize.
