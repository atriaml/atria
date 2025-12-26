from __future__ import annotations

from collections import OrderedDict
from typing import Any, Self

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

BaselineType = torch.Tensor | tuple[torch.Tensor]


def _to_numpy(
    v: torch.Tensor
    | OrderedDict[str, torch.Tensor]
    | list[OrderedDict[str, torch.Tensor]]
    | tuple[torch.Tensor | Any, ...]
    | None,
):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, OrderedDict):
        return OrderedDict({k: val.detach().cpu().numpy() for k, val in v.items()})
    if isinstance(v, list):
        return [
            OrderedDict({k: val.detach().cpu().numpy() for k, val in item.items()})
            for item in v
        ]
    if isinstance(v, tuple):
        return tuple(
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
            for item in v
        )
    return v


def _from_numpy(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    if isinstance(v, OrderedDict):
        return OrderedDict({k: torch.from_numpy(val) for k, val in v.items()})
    if isinstance(v, list):
        return [
            OrderedDict({k: torch.from_numpy(val) for k, val in item.items()})
            for item in v
        ]
    if isinstance(v, tuple):
        return tuple(
            torch.from_numpy(item) if isinstance(item, np.ndarray) else item
            for item in v
        )
    return v


def _to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, OrderedDict):
        return OrderedDict({k: _to_device(v, device) for k, v in obj.items()})
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def _verify_batch_size(
    tensor_or_item_list: torch.Tensor | tuple[torch.Tensor, ...] | list | None,
    expected_batch_size: int,
    batch_dim: int = 0,
):
    if tensor_or_item_list is None:
        return
    if isinstance(tensor_or_item_list, tuple):
        for i, t in enumerate(tensor_or_item_list):
            if isinstance(t, torch.Tensor):
                assert t.shape[batch_dim] == expected_batch_size, (
                    f"All tensors must have the same batch size for unbatching. Tensor {i} has batch size {t.shape[batch_dim]}, expected {expected_batch_size}."
                )
    elif isinstance(tensor_or_item_list, list):
        assert len(tensor_or_item_list) == expected_batch_size, (
            f"All lists must have the same length for unbatching. List has length {len(tensor_or_item_list)}, expected {expected_batch_size}."
        )
    else:
        assert tensor_or_item_list.shape[batch_dim] == expected_batch_size, (
            f"All tensors must have the same batch size for unbatching. Tensor has batch size {tensor_or_item_list.shape[batch_dim]}, expected {expected_batch_size}."
        )


def _extract_sample_from_batch(input: tuple[torch.Tensor, ...] | None, sample_idx: int):
    if input is None:
        return None
    return tuple(
        input_tensor[sample_idx].unsqueeze(0)
        if isinstance(input_tensor, torch.Tensor)
        else input_tensor
        for input_tensor in input
    )


class SampleExplanationTarget(BaseModel):
    """Base class for all target types with Pydantic validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: int | tuple[int, ...]
    name: str


class BatchExplanationTarget(BaseModel):
    """Base class for batch explanation targets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: list[int | tuple[int, ...]]
    name: list[str]

    def tolist(self) -> list[SampleExplanationTarget]:
        return [
            SampleExplanationTarget(value=v, name=n)
            for v, n in zip(self.value, self.name, strict=True)
        ]

    @classmethod
    def fromlist(cls, targets: list[SampleExplanationTarget]) -> Self:
        return cls(value=[t.value for t in targets], name=[t.name for t in targets])

    def subset(self, indices: list[int]) -> BatchExplanationTarget:
        return BatchExplanationTarget(
            value=[self.value[i] for i in indices], name=[self.name[i] for i in indices]
        )


SingleExplanationTargetType = BatchExplanationTarget
ExplanationTargetType = BatchExplanationTarget | list[BatchExplanationTarget] | None
