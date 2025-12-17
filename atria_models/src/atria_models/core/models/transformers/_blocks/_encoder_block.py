import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from atria_models.core.models.transformers._encoder_model._config import (
    TransformersEncoderModelConfig,
)
from atria_models.core.models.transformers._modules._attention import (
    MultiHeadSelfAttention,
)
from atria_models.core.models.transformers._outputs import (
    EncoderLayerOutput,
    EncoderOutput,
)


class AttentionOutputFFN(nn.Module):
    def __init__(
        self, hidden_size: int, layer_norm_eps: float, hidden_dropout_prob: float
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self, hidden_state: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layer_norm(hidden_state + input_tensor)
        return hidden_state


class LayerIntermediateFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act):
        super().__init__()

        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.intermediate_act_fn(hidden_state)
        return hidden_state


class LayoutOutputFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
    ):
        super().__init__()

        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self, hidden_state: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layer_norm(hidden_state + input_tensor)
        return hidden_state


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformersEncoderModelConfig):
        super().__init__()

        self.config = config

        self._build_layers()

    def _build_layers(self):
        # Layers
        self.multi_head_attention = MultiHeadSelfAttention(
            config=self.config.attention_config
        )
        self.attention_output_ffn = AttentionOutputFFN(
            hidden_size=self.config.layers_config.hidden_size,
            layer_norm_eps=self.config.layers_config.layer_norm_eps,
            hidden_dropout_prob=self.config.layers_config.hidden_dropout_prob,
        )
        self.intermediate_ffn = LayerIntermediateFFN(
            self.config.layers_config.hidden_size,
            self.config.layers_config.intermediate_size,
            self.config.layers_config.hidden_act,
        )
        self.output_ffn = LayoutOutputFFN(
            self.config.layers_config.hidden_size,
            self.config.layers_config.intermediate_size,
            self.config.layers_config.layer_norm_eps,
            self.config.layers_config.hidden_dropout_prob,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
    ) -> EncoderLayerOutput:
        attention_outputs = self.multi_head_attention(
            hidden_state, attention_mask, head_mask
        )
        hidden_state = self.attention_output_ffn(
            attention_outputs.context_output, hidden_state
        )
        hidden_state = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.config.layers_config.chunk_size_feed_forward,
            1,  # (B, Seq Len, H)
            hidden_state,
        )
        return EncoderLayerOutput(
            hidden_state=hidden_state, attentions=attention_outputs.attentions
        )

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate_ffn(attention_output)
        hidden_state = self.output_ffn(intermediate_output, attention_output)
        return hidden_state


class Encoder(nn.Module):
    def __init__(self, config: TransformersEncoderModelConfig):
        super().__init__()

        self.config = config
        self._build_layers()

    def _build_layers(self):
        self.layers = nn.ModuleList(
            [
                EncoderLayer(config=self.config)
                for _ in range(self.config.layers_config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
    ) -> EncoderOutput:
        all_hidden_states = ()
        all_attentions = ()
        last_hidden_state = hidden_state

        for i, layer_module in enumerate(self.layers):
            if self.config.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_state, attention_mask, layer_head_mask)

            last_hidden_state = layer_outputs.hidden_state
            print(
                "hidden_states",
                i,
                layer_outputs.hidden_state.shape,
                layer_outputs.hidden_state.sum(),
            )
            if self.config.output_attentions:
                all_attentions = all_attentions + (layer_outputs.attentions,)

        if self.config.output_hidden_states:
            all_hidden_states = all_hidden_states + (last_hidden_state,)

        return EncoderOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
