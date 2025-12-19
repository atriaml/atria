from collections import OrderedDict
from dataclasses import dataclass

import torch
from atria_models.core.models.transformers._configs._encoder_model import (
    EmbeddingsConfig,
)
from torch import nn


@dataclass(frozen=True)
class TokenEmbeddingOutputs:
    token_embeddings: torch.Tensor | None = None
    position_embeddings: torch.Tensor | None = None
    token_type_embeddings: torch.Tensor | None = None

    def sum(self) -> torch.Tensor:
        total = self.token_embeddings + self.position_embeddings
        if self.token_type_embeddings is not None:
            total = total + self.token_type_embeddings
        return total

    def to_ordered_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            token_embeddings=self.token_embeddings,
            position_embeddings=self.position_embeddings,
            token_type_embeddings=self.token_type_embeddings,
        )


class TokenEmbeddings(nn.Module):
    def __init__(self, config: EmbeddingsConfig):
        super().__init__()
        self._config = config
        self._build_model()

    def _build_model(self):
        self._build_embeddings()
        self._build_buffers()

    def _build_embeddings(self):
        self.word_embeddings = nn.Embedding(
            self._config.vocab_size,
            self._config.hidden_size,
            padding_idx=self._config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            self._config.max_position_embeddings, self._config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            self._config.type_vocab_size, self._config.hidden_size
        )

    def _build_buffers(self):
        self.register_buffer(
            "position_ids",
            torch.arange(self._config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros((1, self._config.max_position_embeddings), dtype=torch.long),
            persistent=False,
        )

    def _default_position_ids(
        self, seq_length: int, past_kv_length: int
    ) -> torch.LongTensor:
        # load from buffer and slice
        return self.position_ids[:, past_kv_length : past_kv_length + seq_length]

    def _default_token_type_ids(
        self, batch_size: int, seq_length: int
    ) -> torch.LongTensor:
        # load from buffer, slice and expand
        token_type_ids = self.token_type_ids[:, :seq_length]
        return token_type_ids.expand(batch_size, seq_length)

    def forward(
        self,
        token_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_kv_length: int = 0,
    ) -> TokenEmbeddingOutputs:
        # input shape
        batch_size, seq_length = token_ids.size()

        # resolve ids
        if position_ids is None:
            position_ids = self._default_position_ids(seq_length, past_kv_length)
        if token_type_ids is None:
            token_type_ids = self._default_token_type_ids(batch_size, seq_length)

        # embeddings
        return TokenEmbeddingOutputs(
            token_embeddings=self.word_embeddings(token_ids),
            position_embeddings=self.position_embeddings(position_ids),
            token_type_embeddings=self.token_type_embeddings(token_type_ids),
        )
