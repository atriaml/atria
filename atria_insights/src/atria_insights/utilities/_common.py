from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _map_tensor_tuples_to_keys(
    tensor_tuple: tuple[torch.Tensor, ...], keys: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    return {key: tensor_tuple[idx] for idx, key in enumerate(keys)}


# map back arrays to feature keys
def _map_tensor_dicts_to_tuples(
    tensor_tuple: dict[str, torch.Tensor] | torch.Tensor, keys: tuple[str, ...]
) -> tuple[torch.Tensor, ...]:
    import torch

    if isinstance(tensor_tuple, torch.Tensor):
        return (tensor_tuple,)
    return tuple(tensor_tuple[feature_key] for feature_key in keys)


def _to_device(obj, device):
    import torch

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
