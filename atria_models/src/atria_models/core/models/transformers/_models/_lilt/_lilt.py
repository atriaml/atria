from __future__ import annotations

import torch
from atria_logger import get_logger

from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_models.core.models.transformers._models._lilt._config import (
    LiLTEncoderModelConfig,
)
from atria_models.core.models.transformers._models._lilt._embeddings import (
    LiLTEmbeddingOutputs,
    LiLTEmbeddings,
    LiLTEmbeddingsPostProcessor,
)
from atria_models.core.models.transformers._models._lilt._encoder_block import (
    LiLTEncoderBlock,
)
from atria_models.core.models.transformers._outputs import (
    TransformersEncoderModelOutput,
)
from atria_models.core.models.transformers._utilities import _resolve_head_mask
from atria_models.registry.registry_groups import MODELS

logger = get_logger(__name__)


@MODELS.register("lilt-roberta-base")
class LiLTEncoderModel(TransformersEncoderModel[LiLTEncoderModelConfig]):
    __config__ = LiLTEncoderModelConfig

    def _build_text_layout_embeddings(self):
        return LiLTEmbeddings(self._config.embeddings_config)

    def _build_encoder(self):
        return LiLTEncoderBlock(config=self.config)

    def _build_embeddings_postprocessor(self):
        return LiLTEmbeddingsPostProcessor(
            hidden_size=self.config.layers_config.hidden_size,
            layer_norm_eps=self.config.layers_config.layer_norm_eps,
            dropout_prob=self.config.layers_config.hidden_dropout_prob,
            channel_shrink_ratio=self.config.embeddings_config.channel_shrink_ratio,
        )

    def _resolve_embeddings(  # type: ignore[override]
        self,
        token_ids_or_embeddings: torch.Tensor,
        position_ids_or_embeddings: torch.Tensor | None,
        token_type_ids_or_embeddings: torch.Tensor | None,
        layout_ids_or_embeddings: torch.Tensor | None,
        is_embedding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not is_embedding:
            embeddings = self.ids_to_embeddings(
                token_ids=token_ids_or_embeddings,
                position_ids=position_ids_or_embeddings,
                token_type_ids=token_type_ids_or_embeddings,
                layout_ids=layout_ids_or_embeddings,
            )
        else:
            embeddings = LiLTEmbeddingOutputs(
                token_embeddings=token_ids_or_embeddings,
                position_embeddings=position_ids_or_embeddings,
                token_type_embeddings=token_type_ids_or_embeddings,
                layout_embeddings=layout_ids_or_embeddings,
            )
        return self.embeddings_postprocessor(embeddings=embeddings)

    def ids_to_embeddings(  # type: ignore[override]
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        layout_ids: torch.Tensor | None = None,
    ) -> LiLTEmbeddingOutputs:
        return self.embeddings(
            token_ids=token_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            layout_ids=layout_ids,
        )

    def forward(  # type: ignore[override]
        self,
        token_ids_or_embeddings: torch.Tensor,
        position_ids_or_embeddings: torch.Tensor | None = None,
        token_type_ids_or_embeddings: torch.Tensor | None = None,
        layout_ids_or_embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        is_embedding: bool = False,
        **head_kwargs,
    ) -> TransformersEncoderModelOutput:
        hidden_state, layout_hidden_state = self._resolve_embeddings(
            token_ids_or_embeddings=token_ids_or_embeddings,
            layout_ids_or_embeddings=layout_ids_or_embeddings,
            position_ids_or_embeddings=position_ids_or_embeddings,
            token_type_ids_or_embeddings=token_type_ids_or_embeddings,
            is_embedding=is_embedding,
        )
        head_mask = _resolve_head_mask(
            head_mask, self.config.layers_config.num_hidden_layers, self.dtype
        )
        encoder_outputs = self.encoder(
            hidden_state=hidden_state,
            layout_hidden_state=layout_hidden_state,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        head_output = None
        if self.head is not None:
            head_output = self._head_forward(last_hidden_state, **head_kwargs)
        return TransformersEncoderModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            head_output=head_output,
        )
