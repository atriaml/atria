import torch
from torch import nn


class BertLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[torch.FloatTensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, past_key_value, output_attentions
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        present_key_value = ()
        if self.is_decoder:
            present_key_value = self_attention_outputs[-1]

        # cross-attention
        if encoder_hidden_states is not None:
            if not self.config.add_cross_attention:
                raise ValueError()


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        config,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
    ) -> EncoderTransformerModelOutput:
        # Initialize output containers
        outputs = self._initialize_outputs()

        # Validate cache usage with gradient checkpointing
        use_cache = self._validate_cache_usage()

        # Process through transformer layers
        hidden_states, outputs = self._process_layers(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            outputs,
        )

        # Add final hidden state if needed
        if self.output_hidden_states:
            outputs["all_hidden_states"] = outputs["all_hidden_states"] + (
                hidden_states,
            )

        return EncoderTransformerModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=outputs["next_decoder_cache"],
            hidden_states=outputs["all_hidden_states"],
            attentions=outputs["all_self_attentions"],
            cross_attentions=outputs["all_cross_attentions"],
        )

    def _initialize_outputs(self) -> dict:
        """Initialize output containers based on configuration."""
        return {
            "all_hidden_states": () if self.output_hidden_states else None,
            "all_self_attentions": () if self.output_attentions else None,
            "all_cross_attentions": (
                ()
                if self.output_attentions and self.config.add_cross_attention
                else None
            ),
            "next_decoder_cache": () if self.use_cache else None,
        }

    def _validate_cache_usage(self) -> bool:
        """Validate and adjust cache usage based on gradient checkpointing."""
        if self.gradient_checkpointing and self.training and self.use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            return False
        return self.use_cache

    def _process_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None,
        head_mask: torch.FloatTensor | None,
        encoder_hidden_states: torch.FloatTensor | None,
        encoder_attention_mask: torch.FloatTensor | None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None,
        use_cache: bool,
        outputs: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Process input through all transformer layers."""
        for i, layer_module in enumerate(self.layer):
            # Store intermediate hidden states
            if self.output_hidden_states:
                outputs["all_hidden_states"] = outputs["all_hidden_states"] + (
                    hidden_states,
                )

            # Prepare layer inputs
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # Forward through layer
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=self.output_attentions,
            )

            # Update states and collect outputs
            hidden_states, outputs = self._collect_layer_outputs(
                layer_outputs, use_cache, outputs
            )

        return hidden_states, outputs

    def _collect_layer_outputs(
        self, layer_outputs: tuple, use_cache: bool, outputs: dict
    ) -> tuple[torch.Tensor, dict]:
        """Collect and organize outputs from a single layer."""
        hidden_states = layer_outputs[0]

        # Cache key-value pairs
        if use_cache:
            outputs["next_decoder_cache"] += (layer_outputs[-1],)

        # Collect attention weights
        if self.output_attentions:
            outputs["all_self_attentions"] = outputs["all_self_attentions"] + (
                layer_outputs[1],
            )
            if self.config.add_cross_attention:
                outputs["all_cross_attentions"] = outputs["all_cross_attentions"] + (
                    layer_outputs[2],
                )

        return hidden_states, outputs
