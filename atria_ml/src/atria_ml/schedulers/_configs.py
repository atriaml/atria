from __future__ import annotations

from typing import TYPE_CHECKING

from atria_ml.schedulers._base import LRSchedulerConfig

if TYPE_CHECKING:
    import torch
    from torch.optim.optimizer import Optimizer as Optimizer


class StepLRSchedulerConfig(LRSchedulerConfig):
    module_path: str | None = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 30
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


class MultiStepLRSchedulerConfig(LRSchedulerConfig):
    module_path: str | None = "torch.optim.lr_scheduler.MultiStepLR"
    milestones: list[int] = [30, 80]
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


class ExponentialLRSchedulerConfig(LRSchedulerConfig):
    module_path: str | None = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: float = 0.9
    last_epoch: int = -1
    verbose: bool = False


class CyclicLRSchedulerConfig(LRSchedulerConfig):
    module_path: str | None = "torch.optim.lr_scheduler.CyclicLR"
    base_lr: float = 0.001
    max_lr: float = 0.006
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode: str = "triangular"
    gamma: float = 1.0
    scale_fn: None | str = None
    scale_mode: str = "cycle"
    cycle_momentum: bool = True
    base_momentum: float = 0.8
    max_momentum: float = 0.9
    last_epoch: int = -1


class ReduceLROnPlateauSchedulerConfig(LRSchedulerConfig):
    module_path: str | None = "ignite.handlers.ReduceLROnPlateauScheduler"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8
    verbose: bool = False


class CosineAnnealingLRSchedulerConfig(LRSchedulerConfig):
    module_path: str | None = "torch.optim.lr_scheduler.CosineAnnealingLR"
    eta_min: float = 0.0
    last_epoch: int = -1
    restarts: bool = False
    verbose: bool = False

    def build(  # type: ignore
        self,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
        total_update_steps: int,
        total_warmup_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        import torch

        if not self.restarts:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_update_steps - total_warmup_steps,
                eta_min=self.eta_min,
                last_epoch=self.last_epoch,
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=steps_per_epoch - total_warmup_steps,
                eta_min=self.eta_min,
                last_epoch=self.last_epoch,
            )


class LambdaLRSchedulerConfig(LRSchedulerConfig):
    module_path: str | None = "torch.optim.lr_scheduler.LambdaLR"
    lambda_fn: str = "linear"
    last_epoch: int = -1

    def _get_lambda_fn(
        self, lambda_fn: str, total_update_steps: int, total_warmup_steps: int
    ):
        if lambda_fn == "linear":

            def linear_lambda_lr(current_step: int):
                if current_step < total_warmup_steps:
                    return float(current_step) / float(max(1, total_warmup_steps))
                return max(
                    0.0,
                    float(total_update_steps - current_step)
                    / float(max(1, total_update_steps - total_warmup_steps)),
                )

            return linear_lambda_lr
        else:
            raise ValueError(f"Unknown lambda_fn: {lambda_fn}")

    def build(
        self,
        optimizer: Optimizer,
        total_update_steps: int = 1000,
        total_warmup_steps: int = 100,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=self._get_lambda_fn(
                lambda_fn=self.lambda_fn,
                total_update_steps=total_update_steps,
                total_warmup_steps=total_warmup_steps,
            ),
            last_epoch=self.last_epoch,
        )


class PolynomialDecayLRSchedulerConfig(LRSchedulerConfig):
    module_path: str | None = (
        "atria_ml.schedulers.polynomial_decay_lr.PolynomialDecayLR"
    )
    max_decay_steps: int = -1
    end_learning_rate: float = 0.0001
    power: float = 1.0

    def build(
        self,
        optimizer: Optimizer,
        total_update_steps: int = 1000,
        total_warmup_steps: int = 100,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        from atria_ml.schedulers._polynomial_decay_lr import PolynomialDecayLR

        max_decay_steps = (
            total_update_steps - total_warmup_steps
            if self.max_decay_steps == -1
            else self.max_decay_steps
        )
        return PolynomialDecayLR(
            optimizer=optimizer,
            max_decay_steps=max_decay_steps,
            end_learning_rate=self.end_learning_rate,
            power=self.power,
        )
