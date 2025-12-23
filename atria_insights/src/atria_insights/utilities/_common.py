from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _map_tensor_tuples_to_keys(
    tensor_tuple: tuple[torch.Tensor, ...], keys: tuple[str, ...]
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (feature_key, tensor_tuple[idx]) for idx, feature_key in enumerate(keys)
    )


# map back arrays to feature keys
def _map_tensor_dicts_to_tuples(
    tensor_tuple: dict[str, torch.Tensor] | torch.Tensor, keys: tuple[str, ...]
) -> tuple[torch.Tensor, ...]:
    import torch

    if isinstance(tensor_tuple, torch.Tensor):
        return (tensor_tuple,)
    return tuple(tensor_tuple[feature_key] for feature_key in keys)
