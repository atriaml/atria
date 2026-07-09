from typing import Any

import torch
from ignite.metrics import Metric
from opacus import PrivacyEngine


class PrivacyLossMetric(Metric):
    def __init__(self, privacy_engine: PrivacyEngine, delta: float) -> None:
        super().__init__()
        self._privacy_engine = privacy_engine
        self._delta = delta

    def reset(self) -> None:
        self._epsilon = -1

    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._epsilon = self._privacy_engine.accountant.get_epsilon(self._delta)

    def compute(self) -> list[dict[str, Any]]:
        return self._epsilon
