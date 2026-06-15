---
title: Transformer Encoder
---

# Transformer Encoder

`TransformersEncoderModel` is Atria's native abstract transformer base class. It is both a PyTorch `nn.Module` and a `ConfigurableModule`, making it fully integrated with Atria's registry and config system — and uniquely capable of supporting all three families of explainability methods.

## Why a custom base?

HuggingFace `AutoModel*` wrappers are powerful but opaque: they don't guarantee a stable embedding-space entry point for gradient attribution, and they don't expose a consistent interface for extracting and mapping attention weights. Timm and torchvision models have no token or embedding concept at all.

`TransformersEncoderModel` solves this by defining an explicit **explainability contract** on top of the standard `nn.Module` forward pass. Any model that subclasses it can be used with gradient-based, perturbation-based, and attention-based explainers without modification.

## Architecture

```
TransformersEncoderModel (abstract)
├── BertEncoderModel         (@MODELS.register("bert-base-uncased"))     text only
├── RoBertaEncoderModel      (@MODELS.register("roberta-base"))          text only
├── LiLTEncoderModel         (@MODELS.register("lilt-roberta-base"))     text + layout
└── LayoutLMv3EncoderModel   (@MODELS.register("layoutlmv3-base"))       text + layout + image
```

The abstract base sets `__abstract__ = True` so it is never instantiated directly. Concrete subclasses register themselves in the `MODELS` registry and inherit all XAI interface methods.

These models are built by `AtriaModelBuilder` — not by `TransformersModelBuilder`. `AtriaModelBuilder` looks up the model name in the `MODELS` registry and constructs the full architecture from its config.

### Document-understanding models

LiLT and LayoutLMv3 extend the base with additional input modalities for document AI tasks (layout analysis, entity labeling, question answering over documents):

**LiLTEncoderModel** processes text tokens alongside bounding-box layout coordinates. It runs two parallel transformer streams — one for text, one for layout — and couples them through cross-attention. The `is_embedding=True` path accepts separate text and layout embedding tensors, enabling gradient attribution over both modalities simultaneously. `attn_feature_ids()` returns `{"token_ids", "layout_ids"}`, so attention-based explainers can produce separate attention maps for text and layout streams.

**LayoutLMv3EncoderModel** is fully multimodal: it combines text tokens, bounding-box layout coordinates, and image patches (via a ViT-style patch embedding) in a single unified encoder. The image patches are embedded separately and concatenated with the text sequence before the transformer layers. For XAI, `attn_feature_ids()` returns `{"token_ids", "image"}`, and `map_attentions_to_feature_space()` handles the token/image split in the attention output, upsampling patch-level attention scores back to pixel resolution.

## Config

`TransformersEncoderModelConfig` composes four sub-configs that mirror the structure of a standard transformer encoder:

| Sub-config | Purpose |
|---|---|
| `embeddings_config: EmbeddingsConfig` | Vocab size, hidden size, max position embeddings, special token IDs (PAD, CLS, SEP, MASK) |
| `attention_config: AttentionConfig` | Number of attention heads, head size, attention dropout |
| `layers_config: LayersConfig` | Number of hidden layers, intermediate size, activation, hidden dropout |
| `head_config` | Task head — discriminated union of `SequenceClassificationHead`, `TokenClassificationHead`, `QuestionAnsweringHead` |
| `output_attentions: bool` | When `True`, the model outputs attention weights at every layer — required for attention-based XAI |
| `output_hidden_states: bool` | When `True`, intermediate hidden states are returned |
| `checkpoint_config: CheckpointConfig` | Pretrained checkpoint path (e.g. `"hf://bert-base-uncased"`) and a `key_mapping` list for remapping HuggingFace weight names to Atria's architecture naming |

```python
class BertEncoderModelConfig(TransformersEncoderModelConfig):
    checkpoint_config: CheckpointConfig = CheckpointConfig(
        pretrained_checkpoint="hf://bert-base-uncased",
        key_mapping=[
            ("encoder.layer.", "encoder.layers."),
            ("LayerNorm", "layer_norm"),
            (".attention.self.", ".multi_head_attention."),
            # ...
        ],
    )
```

The `key_mapping` list handles the name differences between HuggingFace's weight naming and Atria's own layer naming, so pretrained checkpoints can be loaded without modifying the architecture.

## The XAI-aware forward pass

The central design decision is the `is_embedding` flag:

```python
def forward(
    self,
    token_ids_or_embeddings: torch.Tensor,
    is_embedding: bool = False,
    attention_mask: torch.Tensor | None = None,
    token_type_ids: torch.Tensor | None = None,
    ...
) -> TransformersEncoderModelOutput:
```

**Normal inference path** (`is_embedding=False`): token IDs enter the model, `ids_to_embeddings()` converts them to embedding vectors, and the encoder processes those embeddings.

**Gradient attribution path** (`is_embedding=True`): the caller passes embedding tensors directly, bypassing the (non-differentiable) token ID lookup. This is essential for gradient-based attribution methods — Integrated Gradients, DeepLIFT, GradientSHAP — which must differentiate through the full forward pass from embeddings to logits. The `SequencePipeline` explanation pipeline uses this path.

```python
# Gradient explainer computes attributions in embedding space
embeddings = model.ids_to_embeddings(token_ids)
attributions = integrated_gradients.attribute(
    inputs=embeddings,
    target=label,
    additional_forward_args=(True,),  # is_embedding=True
)
```

`ids_to_embeddings(token_ids)` is the explicit conversion step — it separates the discrete lookup from the continuous forward pass.

## XAI interface methods

Attention-based explainers interact with the model through three additional methods:

**`attn_feature_ids() -> dict[str, str]`**

Returns `{"token_ids": "token_ids"}` — declares which model input fields correspond to attention features. The `AttnSequencePipeline` uses this to know which input to map attention scores back to.

**`encoder_output_to_attn_tuples(output: TransformersEncoderModelOutput) -> tuple`**

Extracts the raw per-layer attention weight tensors from the model output. Requires `output_attentions=True` in the config.

**`map_attentions_to_feature_space(attentions: tuple) -> torch.Tensor`**

Aggregates and maps the per-head, per-layer attention weights back to the input token space — producing a single attention score per token that can be compared with gradient-based attributions.

Together these three methods form the **attention explainability contract**: any subclass of `TransformersEncoderModel` automatically supports attention-based explainability without any additional implementation work.

## Explainability summary

| Explainer family | Mechanism | Required flag |
|---|---|---|
| Gradient-based (IG, DeepLIFT, GradientSHAP) | `is_embedding=True` in forward | `output_attentions` not needed |
| Perturbation-based (LIME, KernelSHAP, Occlusion) | Standard forward with token masking | Neither flag required |
| Attention-based (rollout, raw weights) | `encoder_output_to_attn_tuples()` + `map_attentions_to_feature_space()` | `output_attentions=True` in config |

This is the complete explainability interface. No other model type in Atria implements all three — `TransformersEncoderModel` is the only architecture where the full explainability suite is available.

For document-understanding models the same interface extends naturally to multiple modalities: LiLT produces separate attention maps per stream (`token_ids`, `layout_ids`), and LayoutLMv3 maps the unified attention output back to both the token sequence and the original image pixel space.
