from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

from atria_prv.fl._trainers._fl_client_trainer import FLClientOutput


class FLAggregationStrategy(ABC):
    """Streaming accumulator for aggregating client updates into global params.

    Used per round: ``reset()`` once, ``update(output)`` for each client as it
    finishes (so only one client's params need to be resident at a time), then
    ``compute()`` to produce the aggregated global state dict.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulator state; called at the start of each round."""

    @abstractmethod
    def update(self, output: FLClientOutput) -> None:
        """Fold one client's result into the accumulator."""

    @abstractmethod
    def compute(self) -> OrderedDict[str, torch.Tensor]:
        """Return the aggregated global params."""

    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load the state dict if required"""

    @abstractmethod
    def state_dict(self):
        """Return the state dict if required"""


class WeightedFedAvg(FLAggregationStrategy):
    """FedAvg: average client params weighted by each client's local sample count."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._accumulated: OrderedDict[str, torch.Tensor] | None = None
        self._total_samples = 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def update(self, output: FLClientOutput) -> None:
        n = output.num_samples
        self._total_samples += n
        if self._accumulated is None:
            self._accumulated = OrderedDict(
                (k, v * n) for k, v in output.params.items()
            )
        else:
            for k, v in output.params.items():
                self._accumulated[k] += v * n

    def compute(self) -> OrderedDict[str, torch.Tensor]:
        assert self._accumulated is not None and self._total_samples > 0, (
            "Cannot aggregate: no client contributed any samples"
        )
        return OrderedDict(
            (k, v / self._total_samples) for k, v in self._accumulated.items()
        )

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}
