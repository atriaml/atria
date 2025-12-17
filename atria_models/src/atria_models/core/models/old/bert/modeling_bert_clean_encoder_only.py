from dataclasses import dataclass

import torch
from atria_logger import get_logger
from atria_models.core.models.transformers._poolers._pooler import DefaultPooler
from atria_models.core.models.transformers.bert._config import (
    TransformersEncoderModelConfig,
)
from atria_models.core.models.transformers.bert._encoder_block import Encoder
from atria_registry import ConfigurableModule
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig

from atria_models.core.model_builders._constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
from atria_models.core.models.transformers._embeddings._aggregators import (
    EmbeddingsAggregator,
)
from atria_models.core.models.transformers._embeddings._token import (
    TokenEmbeddingOutputs,
    TokenEmbeddings,
)
from atria_models.core.models.transformers._utilities import _resolve_head_mask

logger = get_logger(__name__)


@dataclass(frozen=True)
class BertModelOutput:
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CheckpointLoader:
    """Handles loading and mapping of pretrained model checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        cache_dir: str,
        checkpoint_key_mapping: list[tuple[str, str]] | None = None,
    ):
        self.model = model
        self.cache_dir = cache_dir
        self._rewrite_rules = checkpoint_key_mapping or []

    def load_from_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load checkpoint from local path."""
        if checkpoint_path.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            repo_id = checkpoint_path[5:]  # Remove 'hf://' prefix
            checkpoint_path = hf_hub_download(
                repo_id=repo_id, filename="pytorch_model.bin", cache_dir=self.cache_dir
            )

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        processed_state_dict = self._process_state_dict(state_dict)
        keys = self.model.load_state_dict(processed_state_dict, strict=strict)
        if strict:
            assert len(keys.missing_keys) == 0, f"Missing keys: {keys.missing_keys}"
            assert len(keys.unexpected_keys) == 0, (
                f"Unexpected keys: {keys.unexpected_keys}"
            )
        else:
            if len(keys.missing_keys) > 0:
                logger.warning(f"Warning: Missing keys: {keys.missing_keys}")
            if len(keys.unexpected_keys) > 0:
                logger.warning(f"Warning: Unexpected keys: {keys.unexpected_keys}")

    def _process_state_dict(self, state_dict: dict) -> dict:
        """Process raw state dict by removing prefix, remapping keys, and privatizing."""
        processed_dict = {}

        def _remove_first_component(key: str) -> str:
            """Remove the first component from a dot-separated key."""
            return ".".join(key.split(".")[1:])

        def _remap_key(key: str) -> str:
            """Apply rewrite rules to remap key names."""
            for old, new in self._rewrite_rules:
                if old in key:
                    key = key.replace(old, new)
            return key

        for key, value in state_dict.items():
            # Remove model prefix (e.g., 'bert.')
            processed_key = _remove_first_component(key)
            # Apply rewrite rules
            processed_key = _remap_key(processed_key)

            processed_dict[processed_key] = value

        return processed_dict


class TransformersEncoderModel(
    nn.Module, ConfigurableModule[TransformersEncoderModelConfig]
):
    __config__ = TransformersEncoderModelConfig
    __checkpoint_key_mapping__ = [
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
    ]

    def __init__(
        self, config: TransformersEncoderModelConfig, cache_dir: str | None = None
    ):
        nn.Module.__init__(self)
        ConfigurableModule.__init__(self, config=config)
        self._cache_dir = cache_dir or str(_DEFAULT_ATRIA_MODELS_CACHE_DIR)
        self._build()
        if config.pretrained:
            self._load_checkpoint()

    def __repr__(self):
        return nn.Module.__repr__(self)

    def __str__(self):
        return nn.Module.__str__(self)

    @property
    def checkpoint_loader(self) -> CheckpointLoader:
        """Access to the checkpoint loader for manual checkpoint operations."""
        return CheckpointLoader(
            model=self,
            cache_dir=self._cache_dir,
            checkpoint_key_mapping=self.__checkpoint_key_mapping__,
        )

    @property
    def dtype(self):
        return torch.float32

    def _build_embeddings(self):
        return TokenEmbeddings(config=self.config.embeddings_config)

    def _build_embeddings_aggregator(self):
        return EmbeddingsAggregator(
            hidden_size=self.config.layers_config.hidden_size,
            layer_norm_eps=self.config.layers_config.layer_norm_eps,
            dropout_prob=self.config.layers_config.hidden_dropout_prob,
        )

    def _build_encoder(self):
        return Encoder(config=self.config)

    def _build_pooler(self):
        return DefaultPooler(hidden_size=self.config.layers_config.hidden_size)

    def _build(self):
        self.embeddings = self._build_embeddings()
        self.embeddings_aggregator = self._build_embeddings_aggregator()
        self.encoder = self._build_encoder()
        self.pooler = self._build_pooler()

    def _load_checkpoint(self):
        """Load checkpoint if specified in config."""
        self.checkpoint_loader.load_from_checkpoint(
            self.config.checkpoint_config.pretrained_checkpoint
        )

    def _resolve_embeddings(
        self,
        tokens_ids_or_embedding,
        positions_ids_or_embedding,
        token_types_ids_or_embedding,
        is_embedding,
    ):
        if not is_embedding:
            embeddings = self.ids_to_embeddings(
                token_ids=tokens_ids_or_embedding,
                position_ids=positions_ids_or_embedding,
                token_type_ids=token_types_ids_or_embedding,
            )
            token_embeddings = embeddings.token_embeddings
            position_embeddings = embeddings.position_embeddings
            token_type_embeddings = embeddings.token_type_embeddings
        else:
            token_embeddings = tokens_ids_or_embedding
            position_embeddings = positions_ids_or_embedding
            token_type_embeddings = token_types_ids_or_embedding
        return self.embeddings_aggregator(
            (token_embeddings, position_embeddings, token_type_embeddings)
        )

    def ids_to_embeddings(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> TokenEmbeddingOutputs:
        return self.embeddings(
            token_ids=token_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

    def forward(
        self,
        tokens_ids_or_embedding: torch.Tensor,
        positions_ids_or_embedding: torch.Tensor | None = None,
        token_types_ids_or_embedding: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        is_embedding: bool = False,
    ) -> BertModelOutput:
        hidden_state = self._resolve_embeddings(
            tokens_ids_or_embedding=tokens_ids_or_embedding,
            positions_ids_or_embedding=positions_ids_or_embedding,
            token_types_ids_or_embedding=token_types_ids_or_embedding,
            is_embedding=is_embedding,
        )
        head_mask = _resolve_head_mask(
            head_mask, self.config.layers_config.num_hidden_layers, self.dtype
        )
        encoder_outputs = self.encoder(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = (
            self.pooler(last_hidden_state) if self.pooler is not None else None
        )
        return BertModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
