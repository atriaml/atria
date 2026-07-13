from __future__ import annotations

from collections import OrderedDict

import torch
from atria_logger import get_logger
from opacus.accountants.accountant import IAccountant
from torch.optim import Optimizer

from atria_prv.fl._engines._fl_aggregation import FLAggregationStrategy
from atria_prv.fl._trainers._fl_client_trainer import FLClientOutput

logger = get_logger(__name__)


class FeAmAggregation(FLAggregationStrategy):
    def __init__(
        self,
        optimizer: Optimizer,
        privacy_accountant: IAccountant,
        sample_rate: float,
        noise_multiplier: float,
        target_delta: float,
    ) -> None:
        self._optimizer = optimizer
        self._privacy_accountant = privacy_accountant
        self._sample_rate = sample_rate
        self._noise_multiplier = noise_multiplier
        self._target_delta = target_delta
        self.reset()

    def reset(self) -> None:
        self._accumulated_grads: OrderedDict[str, torch.Tensor] | None = None
        self._total_samples = 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def update(self, output: FLClientOutput) -> None:
        n = output.num_samples
        self._total_samples += n
        if self._accumulated_grads is None:
            self._accumulated_grads = OrderedDict(
                (k, v * n) for k, v in output.grads.items()
            )
        else:
            for k, v in output.grads.items():
                self._accumulated_grads[k] += v * n

    def compute(self) -> OrderedDict[str, torch.Tensor]:
        assert self._accumulated_grads is not None and self._total_samples > 0, (
            "Cannot aggregate: no client contributed any samples"
        )

        averaged_grads = OrderedDict(
            (k, v / self._total_samples) for k, v in self._accumulated_grads.items()
        )
        self._optimizer.zero_grad()

        flat_params = [
            p for group in self._optimizer.param_groups for p in group["params"]
        ]
        flat_grads = list(averaged_grads.values())
        flat_keys = list(averaged_grads.keys())

        assert len(flat_params) == len(flat_grads), (
            f"Optimizer has {len(flat_params)} params but got {len(flat_grads)} grads."
        )

        for param, grad in zip(flat_params, flat_grads, strict=True):
            param.grad = grad.to(param.device)

        self._optimizer.step()
        self._privacy_accountant.step(
            noise_multiplier=self._noise_multiplier, sample_rate=self._sample_rate
        )
        epsilon = self._privacy_accountant.get_epsilon(delta=self._target_delta)
        logger.info(
            f"Privacy spent so far: epsilon={epsilon:.3f}, delta={self._target_delta}"
        )

        # after update
        updated_params = OrderedDict(
            (k, p.detach().clone()) for k, p in zip(flat_keys, flat_params, strict=True)
        )

        self._optimizer.zero_grad()
        return updated_params

    def state_dict(self):
        optimizer_state = self._optimizer.state_dict()

        # move all tensors in optimizer state to cpu before saving, so checkpoints
        # are device-agnostic regardless of where training ran
        for state in optimizer_state["state"].values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

        return {
            "optimizer": optimizer_state,
            "privacy_accountant": self._privacy_accountant.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self._privacy_accountant.load_state_dict(state_dict["privacy_accountant"])
