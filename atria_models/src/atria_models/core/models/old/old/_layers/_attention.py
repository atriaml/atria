import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn as nn


@dataclass
class MultiHeadSelfAttentionOutput:
    context_layer: torch.Tensor
    attention_probs: torch.Tensor | None = None


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        max_position_embeddings: int,
        position_embedding_type: Literal[
            "absolute", "relative_key", "relative_key_query"
        ] = "absolute",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._num_attention_heads = num_attention_heads
        self._attention_probs_dropout_prob = attention_probs_dropout_prob
        self._max_position_embeddings = max_position_embeddings
        self._position_embedding_type = position_embedding_type
        self._attention_head_size = int(self._hidden_size / self._num_attention_heads)
        self._all_head_size = self._num_attention_heads * self._attention_head_size
        self._dtype = dtype

        self._validate_arguments()
        self._build()

    def _validate_arguments(self):
        if self._hidden_size % self._num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self._hidden_size}) is not a multiple of the number of attention "
                f"heads ({self._num_attention_heads})"
            )

    def _build_kqv(self):
        # build the key, query, value projection layers
        self.query = nn.Linear(self._hidden_size, self._all_head_size)
        self.key = nn.Linear(self._hidden_size, self._all_head_size)
        self.value = nn.Linear(self._hidden_size, self._all_head_size)

    def _build(self):
        self._build_kqv()
        self.dropout = nn.Dropout(self._attention_probs_dropout_prob)
        if (
            self._position_embedding_type == "relative_key"
            or self._position_embedding_type == "relative_key_query"
        ):
            self.distance_embedding = nn.Embedding(
                2 * self._max_position_embeddings - 1, self._attention_head_size
            )

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self._num_attention_heads,
            self._attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _expand_attention_mask(
        self, attention_mask: torch.LongTensor
    ) -> torch.LongTensor:
        if attention_mask.dim() == 3:
            return cast(torch.LongTensor, attention_mask[:, None, :, :])
        elif attention_mask.dim() == 2:
            return cast(torch.LongTensor, attention_mask[:, None, None, :])
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )

    def _attention_mask_to_additive_bias(
        self, attention_mask: torch.LongTensor, dtype: torch.dtype
    ) -> torch.Tensor:
        attention_mask_with_dtype = attention_mask.to(dtype=dtype)
        attention_mask_with_dtype = (1.0 - attention_mask_with_dtype) * torch.finfo(
            dtype
        ).min
        return attention_mask_with_dtype

    def _reshape_output(self, context_layer: torch.Tensor) -> torch.Tensor:
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self._hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
    ) -> MultiHeadSelfAttentionOutput:
        # Project the queries, keys and values
        key_state = self._transpose_for_scores(self.key(hidden_states))
        value_state = self._transpose_for_scores(self.value(hidden_states))
        query_state = self._transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_state, key_state.transpose(-1, -2))

        # Scale the attention scores as in the original Transformer paper
        attention_scores = attention_scores / math.sqrt(self._attention_head_size)

        # Apply the attention mask (0, 1) -> (-inf, 0)
        if attention_mask is not None:
            attention_mask = self._expand_attention_mask(attention_mask)
            attention_additive_bias = self._attention_mask_to_additive_bias(
                attention_mask, dtype=attention_scores.dtype
            )
            attention_scores = attention_scores + attention_additive_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # generate the context layer
        context_layer = torch.matmul(attention_probs, value_state)

        # reshape the context layer
        context_layer = self._reshape_output(context_layer)

        return MultiHeadSelfAttentionOutput(
            context_layer=context_layer,
            attention_probs=attention_probs if output_attentions else None,
        )
