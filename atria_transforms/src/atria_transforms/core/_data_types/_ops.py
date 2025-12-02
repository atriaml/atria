"""Tensor operations for TensorDataModel instances."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ._base import TensorDataModel


class TensorOperations:
    """Provides tensor operations for TensorDataModel instances."""

    def __init__(self, model: TensorDataModel):
        self.model = model

    @property
    def model_fields(self):
        return self.model.__class__.model_fields

    def _map_tensors(self, fn: Callable) -> TensorDataModel:
        """Apply function to all tensor fields."""
        updates = {}
        for field_name in self.model_fields.keys():
            if field_name == "metadata":
                continue
            val = getattr(self.model, field_name)
            updates[field_name] = fn(val) if isinstance(val, torch.Tensor) else val

        return self.model.model_copy(update=updates)

    def to(self, device: torch.device) -> TensorDataModel:
        return self._map_tensors(lambda t: t.to(device))

    def cpu(self) -> TensorDataModel:
        return self._map_tensors(lambda t: t.cpu())

    def cuda(self) -> TensorDataModel:
        return self._map_tensors(lambda t: t.cuda())

    def numpy(self) -> TensorDataModel:
        return self._map_tensors(lambda t: t.detach().cpu().numpy())
