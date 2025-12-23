# from __future__ import annotations

# from abc import abstractmethod
# from typing import Any

# import torch


# class SequenceModelExplainabilityAdaptor:
#     def __init__(self, model: torch.nn.Module, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.model = model

#     def forward(
#         self,
#         token_ids: torch.Tensor,
#         position_ids: torch.Tensor | None = None,
#         token_type_ids: torch.Tensor | None = None,
#         attention_mask: torch.Tensor | None = None,
#         head_mask: torch.Tensor | None = None,
#         **kwargs,
#     ) -> tuple[dict[str, torch.Tensor], Any]:
#         with torch.no_grad():
#             embeddings = self._ids_to_embeddings(
#                 token_ids=token_ids,
#                 position_ids=position_ids,
#                 token_type_ids=token_type_ids,
#             )
#         return embeddings, self._forward_with_embeddings(
#             **embeddings, attention_mask=attention_mask, head_mask=head_mask
#         )

#     @abstractmethod
#     def _ids_to_embeddings(
#         self,
#         token_ids: torch.Tensor,
#         position_ids: torch.Tensor | None = None,
#         token_type_ids: torch.Tensor | None = None,
#         **kwargs,
#     ) -> dict[str, torch.Tensor]:
#         pass

#     @abstractmethod
#     def _forward_with_embeddings(
#         self,
#         token_embedding: torch.Tensor,
#         position_embedding: torch.Tensor | None = None,
#         token_types_embedding: torch.Tensor | None = None,
#         attention_mask: torch.Tensor | None = None,
#         head_mask: torch.Tensor | None = None,
#         **kwargs,
#     ) -> Any:
#         pass


# class DocumentModelExplainabilityAdaptor:
#     def __init__(self, model: torch.nn.Module, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.model = model

#     def forward(
#         self,
#         token_ids: torch.Tensor,
#         position_ids: torch.Tensor | None = None,
#         token_type_ids: torch.Tensor | None = None,
#         bbox: torch.Tensor | None = None,
#         pixel_values: torch.Tensor | None = None,
#         attention_mask: torch.Tensor | None = None,
#         head_mask: torch.Tensor | None = None,
#         **kwargs,
#     ) -> tuple[dict[str, torch.Tensor], Any]:
#         with torch.no_grad():
#             embeddings = self._ids_to_embeddings(
#                 token_ids=token_ids,
#                 position_ids=position_ids,
#                 token_type_ids=token_type_ids,
#                 bbox=bbox,
#                 pixel_values=pixel_values,
#             )
#         return embeddings, self._forward_with_embeddings(
#             **embeddings, attention_mask=attention_mask, head_mask=head_mask
#         )

#     @abstractmethod
#     def _ids_to_embeddings(
#         self,
#         token_ids: torch.Tensor,
#         position_ids: torch.Tensor | None = None,
#         token_type_ids: torch.Tensor | None = None,
#         bbox: torch.Tensor | None = None,
#         pixel_values: torch.Tensor | None = None,
#         **kwargs,
#     ) -> dict[str, torch.Tensor]:
#         pass

#     @abstractmethod
#     def _forward_with_embeddings(
#         self,
#         token_embedding: torch.Tensor,
#         position_embedding: torch.Tensor | None = None,
#         token_types_embedding: torch.Tensor | None = None,
#         bbox_embedding: torch.Tensor | None = None,
#         bpixel_values_embedding: torch.Tensor | None = None,
#         attention_mask: torch.Tensor | None = None,
#         head_mask: torch.Tensor | None = None,
#         **kwargs,
#     ) -> Any:
#         pass
