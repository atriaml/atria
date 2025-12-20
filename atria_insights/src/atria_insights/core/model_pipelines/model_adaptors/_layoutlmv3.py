# from __future__ import annotations

# from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
#     LayoutLMv3ForQuestionAnswering,
#     LayoutLMv3ForSequenceClassification,
#     LayoutLMv3ForTokenClassification,
# )

# from atria_insights.core.model_pipelines.model_adaptors._sequence import (
#     DocumentModelExplainabilityAdaptor,
# )


# class LayoutLMv3ModelAdaptor(DocumentModelExplainabilityAdaptor):
#     def __init__(
#         self,
#         model: LayoutLMv3ForSequenceClassification
#         | LayoutLMv3ForTokenClassification
#         | LayoutLMv3ForQuestionAnswering,
#     ) -> None:
#         super().__init__(model=model)

#     def _ids_to_embeddings(
#         self,
#         token_ids: torch.Tensor,
#         position_ids: torch.Tensor | None = None,
#         token_type_ids: torch.Tensor | None = None,
#         bbox: torch.Tensor | None = None,
#         pixel_values: torch.Tensor | None = None,
#         **kwargs,
#     ) -> dict[str, torch.Tensor]:
#         embeddings = self.model.layoutlmv3.embeddings(
#             input_ids=token_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             bbox=bbox,
#             pixel_values=pixel_values,
#         )
#         return {
#             "tokens_embedding": embeddings.word_embeddings,
#             "positions_embedding": embeddings.position_embeddings,
#             "token_types_embedding": embeddings.token_type_embeddings,
#             "bbox_embedding": embeddings.token_position_embeddings,
#             "pixel_values_embedding": embeddings.patch_embeddings,
#         }

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
