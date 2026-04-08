"""Tensor operations for TensorDataModel instances."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from ._base import TensorDataModel


class TensorOperations:
    """Provides tensor operations for TensorDataModel instances."""

    def __init__(self, model: TensorDataModel):
        self.model = model

    @property
    def model_fields(self):
        return self.model.__class__.model_fields

    def _map_fields(
        self, fn: Callable, types: tuple = (np.ndarray,)
    ) -> TensorDataModel:
        """Apply function to all fields matching the given types."""
        updates = {}
        for field_name in self.model_fields.keys():
            if field_name == "metadata":
                continue
            val = getattr(self.model, field_name)
            if val is not None and isinstance(val, types):
                updates[field_name] = fn(val)

        if updates:
            return self.model.model_copy(update=updates)
        return self.model

    def to_tensors(self) -> TensorDataModel:
        """Convert all numpy arrays to torch tensors."""
        import torch

        return self._map_fields(lambda a: torch.from_numpy(a), types=(np.ndarray,))

    def to_numpy(self) -> TensorDataModel:
        """Convert all torch tensors to numpy arrays."""
        import torch

        return self._map_fields(
            lambda t: t.detach().cpu().numpy(), types=(torch.Tensor,)
        )

    def to(self, device: torch.device) -> TensorDataModel:
        import torch

        return self._map_fields(lambda t: t.to(device), types=(torch.Tensor,))

    def cpu(self) -> TensorDataModel:
        import torch

        return self._map_fields(lambda t: t.cpu(), types=(torch.Tensor,))

    def cuda(self) -> TensorDataModel:
        import torch

        return self._map_fields(lambda t: t.cuda(), types=(torch.Tensor,))

    def pin_memory(self) -> TensorDataModel:
        import torch

        return self._map_fields(lambda t: t.pin_memory(), types=(torch.Tensor,))
