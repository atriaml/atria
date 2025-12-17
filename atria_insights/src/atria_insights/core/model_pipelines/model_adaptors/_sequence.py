from __future__ import annotations

from abc import abstractmethod

import torch


class SequenceExplainabilityAdaptor:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @abstractmethod
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def position_embeddings(self, position_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def token_type_embeddings(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(
        self,
        input_embeddings: torch.FloatTensor | None,
        position_embeddings: torch.FloatTensor | None,
        token_type_embeddings: torch.FloatTensor | None,
        attention_mask: torch.FloatTensor | None,
        bbox: torch.LongTensor | None,
        labels: torch.LongTensor | None,
    ):
        raise NotImplementedError


class DocumentExplainabilityAdaptor:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @abstractmethod
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def position_embeddings(self, position_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def token_type_embeddings(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def pixel_embeddings(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def bbox_embeddings(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(
        self,
        input_embeddings: torch.FloatTensor | None,
        position_embeddings: torch.FloatTensor | None,
        token_type_embeddings: torch.FloatTensor | None,
        attention_mask: torch.FloatTensor | None,
        bbox: torch.LongTensor | None,
        labels: torch.LongTensor | None,
    ):
        raise NotImplementedError
