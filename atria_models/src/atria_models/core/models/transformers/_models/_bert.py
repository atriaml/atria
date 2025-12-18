from __future__ import annotations

from atria_logger import get_logger

from atria_models.core.models.transformers._configs._common import CheckpointConfig
from atria_models.core.models.transformers._configs._encoder_model import (
    TransformersEncoderModelConfig,
)
from atria_models.core.models.transformers._models._encoder_model import (
    SequenceClassificationModel,
    TransformersEncoderModel,
)

logger = get_logger(__name__)


class BertEncoderModelConfig(TransformersEncoderModelConfig):
    checkpoint_config: CheckpointConfig = CheckpointConfig(
        pretrained_checkpoint="hf://bert-base-uncased",
        key_mapping=[
            # Encoder
            ("encoder.layer.", "encoder.layers."),
            # LayerNorm
            ("LayerNorm", "layer_norm"),
            ("layer_norm.beta", "layer_norm.bias"),
            ("layer_norm.gamma", "layer_norm.weight"),
            # Attention blocks
            (".attention.self.", ".multi_head_attention."),
            (".attention.output.", ".attention_output_ffn."),
            (".intermediate.", ".intermediate_ffn."),
            (".output.", ".output_ffn."),
            # Embeddings
            ("embeddings.layer_norm.", "embeddings_aggregator.layer_norm."),
        ],
    )


class BertEncoderModel(TransformersEncoderModel):
    __config__ = TransformersEncoderModelConfig


class BertSequenceClassificationModel(SequenceClassificationModel):
    __config__ = TransformersEncoderModelConfig


class BertTokenClassification(SequenceClassificationModel):
    __config__ = TransformersEncoderModelConfig


class BertQuestionAnsweringModel(TransformersEncoderModel):
    __config__ = TransformersEncoderModelConfig
