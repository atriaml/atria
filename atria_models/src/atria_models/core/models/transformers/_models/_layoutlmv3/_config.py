from __future__ import annotations

from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict
from transformers.models.layoutlmv3.configuration_layoutlmv3 import (
    LayoutLMv3Config as LiLTEmbeddingsConfig,  # noqa
)

from atria_models.core.models.transformers._configs._common import (
    AttentionConfig,
    CheckpointConfig,
    LayersConfig,
)
from atria_models.core.models.transformers._configs._encoder_model import (
    TransformersEncoderModelConfig,
)
from atria_models.core.models.transformers._models._layoutlmv3._embeddings import (
    LayoutLMv3EmbeddingsConfig,
)

logger = get_logger(__name__)


class LayoutLMv3AttentionConfig(AttentionConfig):
    use_cogview_trick: bool = True
    has_relative_attention_bias: bool = True
    has_spatial_attention_bias: bool = True
    rel_pos_bins: int = 32
    max_rel_pos: int = 128
    rel_2d_pos_bins: int = 64
    max_rel_2d_pos: int = 256


class ImageEmbeddingsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)

    input_size: int = 224
    num_channels: int = 3
    patch_size: int = 16


class LayoutLMv3EncoderModelConfig(TransformersEncoderModelConfig):
    checkpoint_config: CheckpointConfig = CheckpointConfig(
        remove_root_prefix=True,
        pretrained_checkpoint="hf://microsoft/layoutlmv3-base",
        key_mapping=[
            # Encoder
            ("encoder.layer.", "encoder.layers."),
            # # LayerNorm
            ("LayerNorm", "layer_norm"),
            # ("layer_norm.beta", "layer_norm.bias"),
            # ("layer_norm.gamma", "layer_norm.weight"),
            # # Attention blocks
            (".attention.self.", ".multi_head_attention."),
            (".attention.output.", ".attention_output_ffn."),
            (".intermediate.", ".intermediate_ffn."),
            (".output.", ".output_ffn."),
            # Embeddings
            (".cls_token", ".visual_embeddings.cls_token"),
            (".pos_embed", ".visual_embeddings.pos_embed"),
            (".norm.", ".visual_embeddings.image_norm."),
            (".patch_embed.proj.", ".visual_embeddings.proj."),
            (".embeddings.layer_norm.", ".embeddings_postprocessor.layer_norm."),
            (".encoder.rel_pos_bias.", ".attention_bias_module.rel_pos_bias."),
            (".encoder.rel_pos_x_bias.", ".attention_bias_module.rel_pos_x_bias."),
            (".encoder.rel_pos_y_bias.", ".attention_bias_module.rel_pos_y_bias."),
        ],
    )
    embeddings_config: LayoutLMv3EmbeddingsConfig = LayoutLMv3EmbeddingsConfig(  # type: ignore
        vocab_size=50265, pad_token_id=1, type_vocab_size=1, max_position_embeddings=514
    )
    attention_config: LayoutLMv3AttentionConfig = LayoutLMv3AttentionConfig()  # type: ignore
    image_embeddings_config: ImageEmbeddingsConfig = ImageEmbeddingsConfig()
    layers_config: LayersConfig = LayersConfig(layer_norm_eps=1.0e-5)
    text_embed: bool = True
    visual_embed: bool = True
    attach_pooler: bool = False
