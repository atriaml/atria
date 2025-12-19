from __future__ import annotations

from typing import Generic

import torch
from atria_logger import get_logger
from atria_models.core.model_builders._constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
from atria_models.core.models._checkpoint_utilities import CheckpointLoader
from atria_models.core.models.transformers._blocks._encoder_block import EncoderBlock
from atria_models.core.models.transformers._configs._encoder_model import (
    QuestionAnsweringHeadConfig,
    SequenceClassificationHeadConfig,
    T_TransformersEncoderModelConfig,
    TokenClassificationHeadConfig,
    TransformersEncoderModelConfig,
)
from atria_models.core.models.transformers._embeddings._aggregators import (
    EmbeddingsAggregator,
)
from atria_models.core.models.transformers._embeddings._token import (
    TokenEmbeddingOutputs,
    TokenEmbeddings,
)
from atria_models.core.models.transformers._heads._question_answering import (
    QuestionAnsweringHead,
)
from atria_models.core.models.transformers._heads._sequence_classification import (
    SequenceClassificationHead,
)
from atria_models.core.models.transformers._heads._token_classification import (
    TokenClassificationHead,
)
from atria_models.core.models.transformers._modules._pooler import DefaultPooler
from atria_models.core.models.transformers._outputs import (
    TransformersEncoderModelOutput,
)
from atria_models.core.models.transformers._utilities import _resolve_head_mask
from atria_registry import ConfigurableModule
from torch import nn

logger = get_logger(__name__)


class TransformersEncoderModel(
    nn.Module,
    ConfigurableModule[T_TransformersEncoderModelConfig],
    Generic[T_TransformersEncoderModelConfig],
):
    __abstract__ = True

    def __init__(
        self, config: TransformersEncoderModelConfig, cache_dir: str | None = None
    ):
        nn.Module.__init__(self)
        ConfigurableModule.__init__(self, config=config)
        self._cache_dir = cache_dir or str(_DEFAULT_ATRIA_MODELS_CACHE_DIR)
        self._build()
        if config.pretrained:
            self._load_checkpoint()

    @property
    def checkpoint_loader(self) -> CheckpointLoader:
        """Access to the checkpoint loader for manual checkpoint operations."""
        return CheckpointLoader(
            model=self,
            cache_dir=self._cache_dir,
            checkpoint_key_mapping=self.config.checkpoint_config.key_mapping,
        )

    @property
    def dtype(self):
        return torch.float32

    def forward(
        self,
        tokens_ids_or_embedding: torch.Tensor,
        positions_ids_or_embedding: torch.Tensor | None = None,
        token_types_ids_or_embedding: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        is_embedding: bool = False,
        **head_kwargs,
    ) -> TransformersEncoderModelOutput:
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
        pooled_output = self.pooler(last_hidden_state)
        head_output = None
        if self.head is not None:
            head_output = self._head_forward(
                last_hidden_state, pooled_output, **head_kwargs
            )
        return TransformersEncoderModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            head_output=head_output,
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.layers_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.layers_config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _build_embeddings(self):
        return TokenEmbeddings(config=self.config.embeddings_config)

    def _build_embeddings_aggregator(self):
        return EmbeddingsAggregator(
            hidden_size=self.config.layers_config.hidden_size,
            layer_norm_eps=self.config.layers_config.layer_norm_eps,
            dropout_prob=self.config.layers_config.hidden_dropout_prob,
        )

    def _build_encoder(self):
        return EncoderBlock(config=self.config)

    def _build_pooler(self):
        return DefaultPooler(hidden_size=self.config.layers_config.hidden_size)

    def _build_head(self):
        if self.config.head_config is None:
            return None

        if isinstance(self.config.head_config, SequenceClassificationHeadConfig):
            return SequenceClassificationHead(
                num_labels=self.config.head_config.num_labels,
                subtask=self.config.head_config.sub_task,
                hidden_size=self.config.layers_config.hidden_size,
                classifier_dropout=(
                    self.config.layers_config.classifier_dropout
                    if self.config.layers_config.classifier_dropout is not None
                    else self.config.layers_config.hidden_dropout_prob
                ),
            )
        elif isinstance(self.config.head_config, TokenClassificationHeadConfig):
            return TokenClassificationHead(
                num_labels=self.config.head_config.num_labels,
                hidden_size=self.config.layers_config.hidden_size,
                classifier_dropout=(
                    self.config.layers_config.classifier_dropout
                    if self.config.layers_config.classifier_dropout is not None
                    else self.config.layers_config.hidden_dropout_prob
                ),
            )
        elif isinstance(self.config.head_config, QuestionAnsweringHeadConfig):
            return QuestionAnsweringHead(
                hidden_size=self.config.layers_config.hidden_size
            )
        else:
            raise ValueError(
                f"Unsupported head config type: {type(self.config.head_config)}"
            )

    def _build(self):
        self.embeddings = self._build_embeddings()
        self.embeddings_aggregator = self._build_embeddings_aggregator()
        self.encoder = self._build_encoder()
        self.pooler = self._build_pooler()
        self.head = self._build_head()

    def _load_checkpoint(self):
        """Load checkpoint if specified in config."""
        self.checkpoint_loader.load_from_checkpoint(
            self.config.checkpoint_config.pretrained_checkpoint
        )

    def _resolve_embeddings(
        self,
        tokens_ids_or_embedding: torch.Tensor,
        positions_ids_or_embedding: torch.Tensor | None,
        token_types_ids_or_embedding: torch.Tensor | None,
        is_embedding: bool = False,
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
        **kwargs,
    ) -> TokenEmbeddingOutputs:
        return self.embeddings(
            token_ids=token_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

    def _head_forward(
        self,
        last_hidden_state: torch.Tensor,
        pooled_output: torch.Tensor,
        **head_kwargs,
    ):
        if isinstance(self.head, SequenceClassificationHead):
            return self.head(pooled_hidden_state=pooled_output, **head_kwargs)
        elif isinstance(self.head, TokenClassificationHead):
            return self.head(last_hidden_state=last_hidden_state, **head_kwargs)
        elif isinstance(self.head, QuestionAnsweringHead):
            return self.head(last_hidden_state=last_hidden_state, **head_kwargs)
        else:
            raise ValueError(f"Unsupported head type: {type(self.head)}")

    def __repr__(self):
        return nn.Module.__repr__(self)

    def __str__(self):
        return nn.Module.__str__(self)
