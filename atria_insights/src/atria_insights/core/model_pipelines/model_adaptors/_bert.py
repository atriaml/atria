# from __future__ import annotations

# from collections import OrderedDict

# import torch
# from transformers.models.bert.modeling_bert import (
#     BertForQuestionAnswering,
#     BertForSequenceClassification,
#     BertForTokenClassification,
# )

# from atria_insights.core.model_pipelines.model_adaptors._sequence import (
#     SequenceModelExplainabilityAdaptor,
# )


# class BertModelAdaptor(SequenceModelExplainabilityAdaptor):
#     def __init__(
#         self,
#         model: BertForSequenceClassification
#         | BertForTokenClassification
#         | BertForQuestionAnswering,
#     ) -> None:
#         super().__init__(model=model)

#     @property
#     def word_embeddings(self) -> torch.nn.Embedding:
#         return self.model.bert.embeddings.word_embeddings

#     @property
#     def position_embeddings(self) -> torch.nn.Embedding:
#         return self.model.bert.embeddings.position_embeddings

#     @property
#     def token_type_embeddings(self) -> torch.nn.Embedding:
#         return self.model.bert.embeddings.token_type_embeddings

#     @property
#     def position_ids(self) -> torch.Tensor:
#         return self.model.bert.embeddings.position_ids

#     @property
#     def token_type_ids(self) -> torch.Tensor:
#         return self.model.bert.embeddings.token_type_ids

#     def _default_position_ids(self, seq_length: int) -> torch.Tensor:
#         return self.position_ids[:, :seq_length]

#     def _default_token_type_ids(self, batch_size: int, seq_length: int) -> torch.Tensor:
#         token_type_ids = self.token_type_ids[:, :seq_length]
#         return token_type_ids.expand(batch_size, seq_length)

#     def _ids_to_embeddings(
#         self,
#         token_ids: torch.Tensor,
#         position_ids: torch.Tensor | None = None,
#         token_type_ids: torch.Tensor | None = None,
#     ) -> dict[str, torch.Tensor]:
#         # input shape
#         batch_size, seq_length = token_ids.size()

#         # resolve ids
#         if position_ids is None:
#             position_ids = self._default_position_ids(seq_length)
#         if token_type_ids is None:
#             token_type_ids = self._default_token_type_ids(batch_size, seq_length)

#         return OrderedDict(
#             token_embeddings=self.word_embeddings(token_ids),
#             position_embeddings=self.position_embeddings(position_ids),
#             token_type_embeddings=self.token_type_embeddings(token_type_ids),
#         )

#     def _forward_with_embeddings(
#         self,
#         tokens_embedding: torch.Tensor,
#         positions_embedding: torch.Tensor | None = None,
#         token_types_embedding: torch.Tensor | None = None,
#         bbox_embedding: torch.Tensor | None = None,
#         pixel_values_embedding: torch.Tensor | None = None,
#         attention_mask: torch.Tensor | None = None,
#         head_mask: torch.Tensor | None = None,
#         **kwargs,
#     ) -> any:
#         return self.model.layoutlmv3.encoder(
#             inputs_embeds=tokens_embedding,
#             position_embeddings=positions_embedding,
#             token_type_embeddings=token_types_embedding,
#             bbox_embeddings=bbox_embedding,
#             pixel_values_embeddings=pixel_values_embedding,
#             attention_mask=attention_mask,
#             head_mask=head_mask,
#             **kwargs,
#         )
