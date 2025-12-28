from __future__ import annotations

from atria_logger import get_logger

from atria_models.core.models.transformers._configs._common import (
    CheckpointConfig,
    LayersConfig,
)
from atria_models.core.models.transformers._configs._encoder_model import (
    TransformersEncoderModelConfig,
)

from ._embeddings import LiLTEmbeddingsConfig

logger = get_logger(__name__)


class LiLTEncoderModelConfig(TransformersEncoderModelConfig):
    checkpoint_config: CheckpointConfig = CheckpointConfig(
        remove_root_prefix=False,
        pretrained_checkpoint="hf://SCUT-DLVCLab/lilt-roberta-en-base",
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
            # # Layout Attention blocks
            (".attention.layout_self.", ".multi_head_attention."),
            (".attention.layout_output.", ".layout_attention_output_ffn."),
            (".layout_intermediate.", ".layout_intermediate_ffn."),
            (".layout_output.", ".layout_output_ffn."),
            # Embeddings
            (
                "layout_embeddings.x_position_embeddings.",
                "embeddings.layout_embeddings.x_position_embeddings.",
            ),
            (
                "layout_embeddings.y_position_embeddings.",
                "embeddings.layout_embeddings.y_position_embeddings.",
            ),
            (
                "layout_embeddings.h_position_embeddings.",
                "embeddings.layout_embeddings.h_position_embeddings.",
            ),
            (
                "layout_embeddings.w_position_embeddings.",
                "embeddings.layout_embeddings.w_position_embeddings.",
            ),
            (
                "layout_embeddings.box_linear_embeddings.",
                "embeddings.layout_embeddings.box_linear_embeddings.",
            ),
            (
                "layout_embeddings.box_position_embeddings.",
                "embeddings.layout_embeddings.box_position_embeddings.",
            ),
            (
                "layout_embeddings.layer_norm.",
                "embeddings_postprocessor.layout_layer_norm.",
            ),
            ("embeddings.layer_norm.", "embeddings_postprocessor.layer_norm."),
        ],
    )
    embeddings_config: LiLTEmbeddingsConfig = LiLTEmbeddingsConfig(  # type: ignore
        vocab_size=50265, pad_token_id=1, type_vocab_size=1, max_position_embeddings=514
    )
    layers_config: LayersConfig = LayersConfig(layer_norm_eps=1.0e-5)
