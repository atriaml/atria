# import torch
# from pydantic import BaseModel
# from torch import nn


# class EmbeddingConfig(BaseModel):
#     vocab_size: int
#     hidden_size: int
#     pad_token_id: int


# class TokenEmbeddings(nn.Module):
#     def __init__(self, config: EmbeddingConfig):
#         super().__init__()
#         self.embedding = nn.Embedding(
#             config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
#         )

#     def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
#         return self.embedding(input_ids)


# class PositionEmbeddings(nn.Module):
#     def __init__(self, max_position_embeddings: int, hidden_size: int):
#         super().__init__()
#         self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
#         self.register_buffer(
#             "position_ids",
#             torch.arange(max_position_embeddings).unsqueeze(0),
#             persistent=False,
#         )

#     def forward(self, seq_length: int) -> torch.Tensor:
#         return self.embedding(self.position_ids[:, :seq_length])


# class TokenTypeEmbeddings(nn.Module):
#     def __init__(
#         self, type_vocab_size: int, hidden_size: int, max_position_embeddings: int
#     ):
#         super().__init__()
#         self.embedding = nn.Embedding(type_vocab_size, hidden_size)
#         self.register_buffer(
#             "token_type_ids",
#             torch.zeros((1, max_position_embeddings), dtype=torch.long),
#             persistent=False,
#         )

#     def forward(
#         self,
#         batch_size: int,
#         seq_length: int,
#         token_type_ids: torch.LongTensor | None = None,
#     ) -> torch.Tensor:
#         if token_type_ids is None:
#             token_type_ids = self.token_type_ids[:, :seq_length].expand(
#                 batch_size, seq_length
#             )
#         return self.embedding(token_type_ids)


# class EmbeddingsAggregator(nn.Module):
#     def __init__(self, hidden_size: int, layer_norm_eps: float, dropout_prob: float):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
#         x = embeddings[0]
#         for emb in embeddings[1:]:
#             x = x + emb

#         x = self.layer_norm(x)
#         x = self.dropout(x)
#         return x


# class SequenceEmbeddingsConfig(BaseModel):
#     vocab_size: int
#     hidden_size: int
#     pad_token_id: int
#     max_position_embeddings: int
#     type_vocab_size: int
#     layer_norm_eps: float
#     hidden_dropout_prob: float
#     position_embedding_type: str = "absolute"


# class SequenceEmbeddings(nn.Module):
#     def __init__(self, config: SequenceEmbeddingsConfig):
#         super().__init__()
#         self._config = config
#         self._build_model()

#     def _build_model(self):
#         self._build_embeddings()
#         self._build_output_transforms()
#         self._build_buffers()

#     def _build_embeddings(self):
#         self.word_embeddings = nn.Embedding(
#             self._config.vocab_size,
#             self._config.hidden_size,
#             padding_idx=self._config.pad_token_id,
#         )
#         self.position_embeddings = nn.Embedding(
#             self._config.max_position_embeddings, self._config.hidden_size
#         )
#         self.token_type_embeddings = nn.Embedding(
#             self._config.type_vocab_size, self._config.hidden_size
#         )

#     def _build_output_transforms(self):
#         self.layer_norm = nn.LayerNorm(
#             self._config.hidden_size, eps=self._config.layer_norm_eps
#         )
#         self.dropout = nn.Dropout(self._config.hidden_dropout_prob)
#         self.position_embedding_type = getattr(
#             self._config, "position_embedding_type", "absolute"
#         )

#     def _build_buffers(self):
#         self.register_buffer(
#             "position_ids",
#             torch.arange(self._config.max_position_embeddings).unsqueeze(0),
#             persistent=False,
#         )
#         self.register_buffer(
#             "token_type_ids",
#             torch.zeros((1, self._config.max_position_embeddings), dtype=torch.long),
#             persistent=False,
#         )

#     # ----------------------------
#     # helpers
#     # ----------------------------
#     def _resolve_position_ids(
#         self,
#         position_ids: torch.LongTensor | None,
#         seq_length: int,
#         past_key_values_length: int,
#     ) -> torch.LongTensor:
#         if position_ids is not None:
#             return position_ids

#         return self.position_ids[
#             :, past_key_values_length : past_key_values_length + seq_length
#         ]

#     def _resolve_token_type_ids(
#         self, token_type_ids: torch.LongTensor | None, batch_size: int, seq_length: int
#     ) -> torch.LongTensor:
#         if token_type_ids is not None:
#             return token_type_ids

#         token_type_ids = self.token_type_ids[:, :seq_length]
#         return token_type_ids.expand(batch_size, seq_length)

#     # ----------------------------
#     # forward
#     # ----------------------------
#     def forward(
#         self,
#         input_ids: torch.LongTensor | None = None,
#         token_type_ids: torch.LongTensor | None = None,
#         position_ids: torch.LongTensor | None = None,
#         inputs_embeds: torch.FloatTensor | None = None,
#         past_key_values_length: int = 0,
#     ) -> torch.Tensor:
#         if input_ids is not None:
#             batch_size, seq_length = input_ids.size()
#         else:
#             batch_size, seq_length = inputs_embeds.size()[:-1]

#         position_ids = self._resolve_position_ids(
#             position_ids, seq_length, past_key_values_length
#         )
#         token_type_ids = self._resolve_token_type_ids(
#             token_type_ids, batch_size, seq_length
#         )

#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)

#         embeddings = inputs_embeds
#         embeddings += self.token_type_embeddings(token_type_ids)

#         if self.position_embedding_type == "absolute":
#             embeddings += self.position_embeddings(position_ids)

#         embeddings = self.layer_norm(embeddings)
#         embeddings = self.dropout(embeddings)

#         return embeddings
