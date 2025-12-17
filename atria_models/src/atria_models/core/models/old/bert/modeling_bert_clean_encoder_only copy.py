from dataclasses import dataclass

import torch
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


class TransformersEncoderModel(
    nn.Module, ConfigurableModule[TransformersEncoderModelConfig]
):
    __config__ = TransformersEncoderModelConfig

    def __init__(
        self, config: TransformersEncoderModelConfig, cache_dir: str | None = None
    ):
        nn.Module.__init__(self)
        ConfigurableModule.__init__(self, config=config)
        self._cache_dir = cache_dir or _DEFAULT_ATRIA_MODELS_CACHE_DIR
        self._build()
        self._load_checkpoint()

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
        if self.config.checkpoint_config.pretrained_checkpoint is not None:
            assert self.config.checkpoint_config.pretrained_checkpoint.startswith(
                "hf://"
            ), "Only Hugging Face pretrained checkpoints are supported currently."
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                repo_id=self.config.checkpoint_config.pretrained_checkpoint[5:],
                filename="pytorch_model.bin",
                cache_dir=self._cache_dir,
            )
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # remap checkpoint keys to match model architecture
            REWRITE_RULES = [
                # Encoder
                ("encoder.layer.", "encoder.layers."),
                # LayerNorm
                ("LayerNorm", "layer_norm"),
                # Attention blocks
                (".attention.self.output.", ".attention_output_ffn."),
                (".attention.self.", ".multi_head_attention."),
                (".intermediate.", ".intermediate_ffn."),
                (".output.", ".output_ffn."),
            ]

            def remove_first(key: str) -> str:
                return ".".join(key.split(".")[1:])

            def remap_key(key: str) -> str:
                for old, new in REWRITE_RULES:
                    if old in key:
                        key = key.replace(old, new)
                return key

            def privatize(key: str) -> str:
                parts = key.split(".")
                last_idx = len(parts) - 1

                out = []
                for i, p in enumerate(parts):
                    if i == last_idx:  # parameter name (weight, bias, etc.)
                        out.append(p)
                    elif p.isdigit():  # list / layer index
                        out.append(p)
                    elif p.startswith("_"):  # already private
                        out.append(p)
                    else:
                        out.append(f"_{p}")

                return ".".join(out)

            new_state_dict = {
                remap_key(remove_first(k)): v for k, v in state_dict.items()
            }

            self.load_state_dict(new_state_dict, strict=True)

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
