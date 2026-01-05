from __future__ import annotations

import math

import torch
from atria_logger import get_logger
from torch import nn

from atria_models.core.models.transformers._heads._token_classification import (
    TokenClassificationHead,
)
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_models.core.models.transformers._models._layoutlmv3._config import (
    LayoutLMv3AttentionConfig,
    LayoutLMv3EncoderModelConfig,
)
from atria_models.core.models.transformers._models._layoutlmv3._embeddings import (
    LayoutLMv3EmbeddingOutputs,
    LayoutLMv3Embeddings,
)
from atria_models.core.models.transformers._outputs import (
    TransformersEncoderModelOutput,
)
from atria_models.core.models.transformers._utilities import _resolve_head_mask
from atria_models.registry.registry_groups import MODELS

logger = get_logger(__name__)


class LayoutLMv3VisualEmbeddings(nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()

        self._input_size = (
            input_size if isinstance(input_size, tuple) else (input_size, input_size)
        )
        self._patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self._grid_size = (
            self._input_size[0] // self._patch_size[0],
            self._input_size[1] // self._patch_size[1],
        )
        self._num_channels = num_channels
        self._hidden_size = hidden_size
        self._hidden_dropout_prob = hidden_dropout_prob
        self._layer_norm_eps = layer_norm_eps

        self._build_layers()

    def _build_layers(self):
        self.proj = nn.Conv2d(
            self._num_channels,
            self._hidden_size,
            kernel_size=self._patch_size,
            stride=self._patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self._hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, self._grid_size[0] * self._grid_size[1] + 1, self._hidden_size
            )
        )
        self.pos_drop = nn.Dropout(p=0.0)
        self.image_norm = nn.LayerNorm(self._hidden_size, eps=1e-6)

    def forward(self, image: torch.Tensor):
        embeddings = self.proj(image).flatten(2).transpose(1, 2)
        batch_size, _, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed
        embeddings = self.pos_drop(embeddings)
        embeddings = self.image_norm(embeddings)
        return embeddings


class LayoutLMv3AttentionBiasModule(nn.Module):
    def __init__(self, attention_config: LayoutLMv3AttentionConfig):
        super().__init__()

        self._attention_config = attention_config

        self._build_layers()

    def _build_layers(self):
        self.has_relative_attention_bias = (
            self._attention_config.has_relative_attention_bias
        )
        self.has_spatial_attention_bias = (
            self._attention_config.has_spatial_attention_bias
        )

        if self.has_relative_attention_bias:
            self.rel_pos_bins = self._attention_config.rel_pos_bins
            self.max_rel_pos = self._attention_config.max_rel_pos
            self.rel_pos_bias = nn.Linear(
                self.rel_pos_bins,
                self._attention_config.num_attention_heads,
                bias=False,
            )

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = self._attention_config.max_rel_2d_pos
            self.rel_2d_pos_bins = self._attention_config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(
                self.rel_2d_pos_bins,
                self._attention_config.num_attention_heads,
                bias=False,
            )
            self.rel_pos_y_bias = nn.Linear(
                self.rel_2d_pos_bins,
                self._attention_config.num_attention_heads,
                bias=False,
            )

    def relative_position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = torch.abs(relative_position)
        else:
            n = torch.max(-relative_position, torch.zeros_like(relative_position))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = self.relative_position_bucket(
            rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos
        )
        # Since this is a simple indexing operation that is independent of the input,
        # no need to track gradients for this operation
        #
        # Without this no_grad context, training speed slows down significantly
        with torch.no_grad():
            rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(
            -1
        )
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(
            -1
        )
        rel_pos_x = self.relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = self.relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # Since this is a simple indexing operation that is independent of the input,
        # no need to track gradients for this operation
        #
        # Without this no_grad context, training speed slows down significantly
        with torch.no_grad():
            rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
            rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self, position_ids: torch.Tensor | None = None, bbox: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        attention_bias = None
        if self.has_relative_attention_bias and position_ids is not None:
            attention_bias = self._cal_1d_pos_emb(position_ids)
        if self.has_spatial_attention_bias and bbox is not None:
            attention_bias_2d = self._cal_2d_pos_emb(bbox)
            if attention_bias is not None:
                attention_bias += attention_bias_2d
            else:
                attention_bias = attention_bias_2d
        return attention_bias


@MODELS.register("layoutlmv3-base")
class LayoutLMv3EncoderModel(TransformersEncoderModel[LayoutLMv3EncoderModelConfig]):
    __config__ = LayoutLMv3EncoderModelConfig

    def _build_text_layout_embeddings(self):
        return LayoutLMv3Embeddings(self._config.embeddings_config)

    def _build(self):
        super()._build()
        self.visual_embeddings = LayoutLMv3VisualEmbeddings(
            input_size=self.config.image_embeddings_config.input_size,
            patch_size=self.config.image_embeddings_config.patch_size,
            num_channels=self.config.image_embeddings_config.num_channels,
            hidden_size=self.config.layers_config.hidden_size,
            layer_norm_eps=self.config.layers_config.layer_norm_eps,
            hidden_dropout_prob=self.config.layers_config.hidden_dropout_prob,
        )
        if (
            self.config.attention_config.has_relative_attention_bias
            or self.config.attention_config.has_spatial_attention_bias
        ):
            size = (
                self.config.image_embeddings_config.input_size
                // self.config.image_embeddings_config.patch_size
            )
            self._init_visual_bbox(image_size=(size, size))
            self.attention_bias_module = LayoutLMv3AttentionBiasModule(
                attention_config=self.config.attention_config
            )
        self.layer_norm = nn.LayerNorm(
            self.config.layers_config.hidden_size,
            eps=self.config.layers_config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(self.config.layers_config.hidden_dropout_prob)

    def _init_visual_bbox(self, image_size=(14, 14), max_len=1000):
        visual_bbox_x = torch.div(
            torch.arange(0, max_len * (image_size[1] + 1), max_len),
            image_size[1],
            rounding_mode="trunc",
        )
        visual_bbox_y = torch.div(
            torch.arange(0, max_len * (image_size[0] + 1), max_len),
            image_size[0],
            rounding_mode="trunc",
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_size[0], 1),
                visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_size[0], 1),
                visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def _calculate_visual_bbox(self, device, dtype, batch_size):
        visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def _resolve_embeddings(  # type: ignore[override]
        self,
        token_ids_or_embeddings: torch.Tensor,
        position_ids_or_embeddings: torch.Tensor | None,
        token_type_ids_or_embeddings: torch.Tensor | None,
        layout_ids_or_embeddings: torch.Tensor | None,
        is_embedding: bool = False,
    ) -> torch.Tensor:
        if not is_embedding:
            embeddings = self.ids_to_embeddings(
                token_ids=token_ids_or_embeddings,
                position_ids=position_ids_or_embeddings,
                token_type_ids=token_type_ids_or_embeddings,
                layout_ids=layout_ids_or_embeddings,
            )
        else:
            embeddings = LayoutLMv3EmbeddingOutputs(
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
    ) -> LayoutLMv3EmbeddingOutputs:
        return self.embeddings(
            token_ids=token_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            layout_ids=layout_ids,
        )

    def forward(  # type: ignore[override]
        self,
        token_ids_or_embeddings: torch.Tensor | None = None,
        position_ids_or_embeddings: torch.Tensor | None = None,
        token_type_ids_or_embeddings: torch.Tensor | None = None,
        layout_ids_or_embeddings: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        layout_ids: torch.Tensor
        | None = None,  # requires layout ids even if only embeddings are passed
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        is_embedding: bool = False,
        **head_kwargs,
    ) -> TransformersEncoderModelOutput:
        if token_ids_or_embeddings is not None:
            hidden_state = self._resolve_embeddings(
                token_ids_or_embeddings=token_ids_or_embeddings,
                layout_ids_or_embeddings=layout_ids_or_embeddings,
                position_ids_or_embeddings=position_ids_or_embeddings,
                token_type_ids_or_embeddings=token_type_ids_or_embeddings,
                is_embedding=is_embedding,
            )
            batch_size, seq_len, device = (
                hidden_state.shape[0],
                hidden_state.shape[1],
                hidden_state.device,
            )
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_len), device=hidden_state.device
                )
        else:
            assert image is not None, "Either input_ids or image must be provided."
            batch_size = image.shape[0]
            seq_len = 0
            device = image.device
        final_bbox = final_position_ids = None
        if image is not None:
            visual_embeddings = self.visual_embeddings(image)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]),
                dtype=torch.long,
                device=device,
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, visual_attention_mask], dim=1
                )
            else:
                attention_mask = visual_attention_mask

            if (
                self.config.attention_config.has_relative_attention_bias
                or self.config.attention_config.has_spatial_attention_bias
            ):
                if self.config.attention_config.has_spatial_attention_bias:
                    visual_bbox = self._calculate_visual_bbox(
                        device, dtype=torch.long, batch_size=batch_size
                    )
                    if layout_ids is not None:
                        final_bbox = torch.cat([layout_ids, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)

                if token_ids_or_embeddings is not None:
                    position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
                    position_ids = position_ids.expand((batch_size, seq_len))
                    final_position_ids = torch.cat(
                        [position_ids, visual_position_ids], dim=1
                    )
                else:
                    final_position_ids = visual_position_ids

            if token_ids_or_embeddings is not None:
                hidden_state = torch.cat([hidden_state, visual_embeddings], dim=1)
            else:
                hidden_state = visual_embeddings

            hidden_state = self.layer_norm(hidden_state)
            hidden_state = self.dropout(hidden_state)
        else:
            if self.config.attention_config.has_spatial_attention_bias:
                if layout_ids is not None:
                    final_bbox = layout_ids
                else:
                    final_bbox = torch.zeros(
                        (batch_size, seq_len, 4), dtype=torch.long, device=device
                    )
            if self.config.attention_config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, :seq_len]
                position_ids = position_ids.expand((batch_size, seq_len))
                final_position_ids = position_ids

        head_mask = _resolve_head_mask(
            head_mask, self.config.layers_config.num_hidden_layers, self.dtype
        )
        attention_bias = self.attention_bias_module(
            position_ids=final_position_ids, bbox=final_bbox
        )
        encoder_outputs = self.encoder(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            head_mask=head_mask,
            attention_bias=attention_bias,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        head_output = None
        if self.head is not None:
            if isinstance(self.head, TokenClassificationHead):  # noqa: F821
                assert seq_len > 0, "TokenClassificationHead requires token inputs."
                head_output = self._head_forward(
                    last_hidden_state[:, :seq_len, :], **head_kwargs
                )
            else:
                head_output = self._head_forward(last_hidden_state, **head_kwargs)
        return TransformersEncoderModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            head_output=head_output,
        )
