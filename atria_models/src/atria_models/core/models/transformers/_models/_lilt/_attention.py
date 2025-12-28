import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn


@dataclass
class LiLTAttentionOutput:
    context_output: torch.Tensor
    layout_context_output: torch.Tensor
    attentions: torch.Tensor | None = None


class LiLTMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
        hidden_size: int = 768,
        channel_shrink_ratio: int = 4,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.channel_shrink_ratio = channel_shrink_ratio

        self._build_layers()

    @property
    def attention_head_size(self) -> int:
        return int(self.hidden_size / self.num_attention_heads)

    @property
    def all_head_size(self) -> int:
        return self.num_attention_heads * self.attention_head_size

    def _validate_attributes(self):
        if self.all_head_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

    def _build_kqv(self):
        # build the key, query, value projection layers
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.layout_query = nn.Linear(
            self.hidden_size // self.channel_shrink_ratio,
            self.all_head_size // self.channel_shrink_ratio,
        )
        self.layout_key = nn.Linear(
            self.hidden_size // self.channel_shrink_ratio,
            self.all_head_size // self.channel_shrink_ratio,
        )
        self.layout_value = nn.Linear(
            self.hidden_size // self.channel_shrink_ratio,
            self.all_head_size // self.channel_shrink_ratio,
        )

    def _build_layers(self):
        self._build_kqv()
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)

    def _transpose_for_scores(self, x, r=1):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size // r,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _expand_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() == 3:
            return cast(torch.Tensor, attention_mask[:, None, :, :])
        elif attention_mask.dim() == 2:
            return cast(torch.Tensor, attention_mask[:, None, None, :])
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )

    def _attention_mask_to_additive_bias(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        attention_mask_with_dtype = attention_mask.to(dtype=dtype)
        attention_mask_with_dtype = (1.0 - attention_mask_with_dtype) * torch.finfo(
            dtype
        ).min
        return attention_mask_with_dtype

    def _reshape_output(self, context_layer: torch.Tensor) -> torch.Tensor:
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
    ) -> LiLTAttentionOutput:
        attention_additive_bias = None
        if attention_mask is not None:
            attention_mask = self._expand_attention_mask(attention_mask)
            attention_additive_bias = self._attention_mask_to_additive_bias(
                attention_mask, dtype=hidden_states.dtype
            )

        layout_value_layer = self._transpose_for_scores(
            self.layout_value(layout_hidden_states), r=self.channel_shrink_ratio
        )
        layout_key_layer = self._transpose_for_scores(
            self.layout_key(layout_hidden_states), r=self.channel_shrink_ratio
        )
        layout_query_layer = self._transpose_for_scores(
            self.layout_query(layout_hidden_states), r=self.channel_shrink_ratio
        )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self._transpose_for_scores(self.key(hidden_states))
        value_layer = self._transpose_for_scores(self.value(hidden_states))
        query_layer = self._transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        layout_attention_scores = torch.matmul(
            layout_query_layer, layout_key_layer.transpose(-1, -2)
        )

        tmp_attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        tmp_layout_attention_scores = layout_attention_scores / math.sqrt(
            self.attention_head_size // self.channel_shrink_ratio
        )
        attention_scores = tmp_attention_scores + tmp_layout_attention_scores
        layout_attention_scores = tmp_layout_attention_scores + tmp_attention_scores

        if attention_additive_bias is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            layout_attention_scores = layout_attention_scores + attention_additive_bias

        # Normalize the attention scores to probabilities.
        layout_attention_probs = nn.Softmax(dim=-1)(layout_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        layout_attention_probs = self.dropout(layout_attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            layout_attention_probs = layout_attention_probs * head_mask

        layout_context_layer = torch.matmul(layout_attention_probs, layout_value_layer)

        layout_context_layer = layout_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = layout_context_layer.size()[:-2] + (
            self.all_head_size // self.channel_shrink_ratio,
        )
        layout_context_layer = layout_context_layer.view(*new_context_layer_shape)

        if attention_additive_bias is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_additive_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return LiLTAttentionOutput(
            context_output=context_layer,
            layout_context_output=layout_context_layer,
            attentions=attention_probs,
        )
