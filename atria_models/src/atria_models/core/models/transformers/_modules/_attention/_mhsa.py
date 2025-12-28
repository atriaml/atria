import math
from typing import cast

import torch
import torch.nn as nn

from atria_models.core.models.transformers._configs._common import AttentionConfig
from atria_models.core.models.transformers._outputs import AttentionOutput


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()

        self._config = config

        self._build_layers()

    def _validate_attributes(self):
        if self._config.all_head_size % self._config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self._config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self._config.num_attention_heads})"
            )

    def _build_kqv(self):
        # build the key, query, value projection layers
        self.query = nn.Linear(self._config.hidden_size, self._config.all_head_size)
        self.key = nn.Linear(self._config.hidden_size, self._config.all_head_size)
        self.value = nn.Linear(self._config.hidden_size, self._config.all_head_size)

    def _build_layers(self):
        self._build_kqv()
        self.dropout = nn.Dropout(self._config.attention_probs_dropout_prob)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self._config.num_attention_heads,
            self._config.attention_head_size,
        )
        x = x.view(new_x_shape)
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
        new_context_layer_shape = context_layer.size()[:-2] + (
            self._config.hidden_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

    def _cogview_attention(self, attention_scores, alpha=32):
        """
        https://huggingface.co/papers/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original nn.Softmax(dim=-1)(attention_scores). Seems the new attention_probs
        will result in a slower speed and a little bias. Can use torch.allclose(standard_attention_probs,
        cogview_attention_probs, atol=1e-08) for comparison. The smaller atol (e.g., 1e-08), the better.
        """
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.Softmax(dim=-1)(new_attention_scores)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> AttentionOutput:
        # Project the queries, keys and values
        key_state = self._transpose_for_scores(self.key(hidden_state))
        value_state = self._transpose_for_scores(self.value(hidden_state))
        query_state = self._transpose_for_scores(self.query(hidden_state))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_state, key_state.transpose(-1, -2))

        if attention_bias is not None:
            attention_scores += attention_bias

        # Scale the attention scores as in the original Transformer paper
        attention_scores = attention_scores / math.sqrt(
            self._config.attention_head_size
        )

        # Apply the attention mask (0, 1) -> (-inf, 0)
        if attention_mask is not None:
            attention_mask = self._expand_attention_mask(attention_mask)
            attention_additive_bias = self._attention_mask_to_additive_bias(
                attention_mask, dtype=hidden_state.dtype
            )
            attention_scores = attention_scores + attention_additive_bias

        # Normalize the attention scores to probabilities.
        if self._config.use_cogview_trick:
            attentions = self._cogview_attention(attention_scores)
        else:
            attentions = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attentions = self.dropout(attentions)

        # Mask heads if we want to
        if head_mask is not None:
            attentions = attentions * head_mask

        # generate the context layer
        context_output = torch.matmul(attentions, value_state)

        # reshape the context layer
        context_output = self._reshape_output(context_output)

        return AttentionOutput(context_output=context_output, attentions=attentions)
