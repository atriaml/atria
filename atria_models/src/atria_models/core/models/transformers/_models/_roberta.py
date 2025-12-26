from __future__ import annotations

import torch
from atria_logger import get_logger

from atria_models.core.models.transformers._configs._common import (
    CheckpointConfig,
    EmbeddingsConfig,
    LayersConfig,
)
from atria_models.core.models.transformers._configs._encoder_model import (
    TransformersEncoderModelConfig,
)
from atria_models.core.models.transformers._embeddings._token import (
    TokenEmbeddingOutputs,
    TokenEmbeddings,
)
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_models.registry.registry_groups import MODELS

logger = get_logger(__name__)


class RoBertaEncoderModelConfig(TransformersEncoderModelConfig):
    checkpoint_config: CheckpointConfig = CheckpointConfig(
        pretrained_checkpoint="hf://roberta-base",
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
    embeddings_config: EmbeddingsConfig = EmbeddingsConfig(
        vocab_size=50265, pad_token_id=1, type_vocab_size=1, max_position_embeddings=514
    )
    layers_config: LayersConfig = LayersConfig(layer_norm_eps=1.0e-5)


class RoBertaTokenEmbeddings(TokenEmbeddings):
    # replicated from https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modular_roberta.py
    def _build_embeddings(self):
        from torch import nn

        self.word_embeddings = nn.Embedding(
            self._config.vocab_size,
            self._config.hidden_size,
            padding_idx=self._config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            self._config.max_position_embeddings,
            self._config.hidden_size,
            padding_idx=self._config.pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(
            self._config.type_vocab_size, self._config.hidden_size
        )

    def _default_position_ids(  # type: ignore[override]
        self, token_ids: torch.Tensor, past_key_values_length: int
    ) -> torch.Tensor:
        # load from buffer and slice
        mask = token_ids.ne(self._config.pad_token_id).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + self._config.pad_token_id

    def _default_token_type_ids(  # type: ignore[override]
        self, position_ids: torch.Tensor, batch_size: int, seq_length: int
    ) -> torch.Tensor:
        if hasattr(self, "token_type_ids"):
            # NOTE: We assume either pos ids to have bsz == 1 (broadcastable) or bsz == effective bsz (input_shape[0])
            buffered_token_type_ids = self.token_type_ids.expand(
                position_ids.shape[0], -1
            )
            buffered_token_type_ids = torch.gather(
                buffered_token_type_ids, dim=1, index=position_ids
            )
            token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        else:
            token_type_ids = torch.zeros(
                (batch_size, seq_length),
                dtype=torch.long,
                device=self.position_ids.device,
            )
        return token_type_ids

    def forward(
        self,
        token_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_kv_length: int = 0,
    ) -> TokenEmbeddingOutputs:
        # input shape
        batch_size, seq_length = token_ids.size()

        # resolve ids
        if position_ids is None:
            position_ids = self._default_position_ids(token_ids, past_kv_length)
        if token_type_ids is None:
            token_type_ids = self._default_token_type_ids(
                position_ids, batch_size, seq_length
            )

        # embeddings
        return TokenEmbeddingOutputs(
            token_embeddings=self.word_embeddings(token_ids),
            position_embeddings=self.position_embeddings(position_ids),
            token_type_embeddings=self.token_type_embeddings(token_type_ids),
        )


@MODELS.register("roberta-base")
class RoBertaEncoderModel(TransformersEncoderModel):
    __config__ = RoBertaEncoderModelConfig

    def _build_embeddings(self):
        return RoBertaTokenEmbeddings(config=self.config.embeddings_config)
