"""
This module defines a custom learning rate scheduler for PyTorch optimizers.

The `polynomial_decay_lr` function registers a learning rate scheduler that applies a polynomial decay to the learning rate over a specified number of training steps.
"""

from typing import Any

from atria_ml.registry import LR_SCHEDULER


@LR_SCHEDULER.register("polynomial_decay_lr")
def polynomial_decay_lr(
    optimizer: Any, total_update_steps: int, max_decay_steps: int = -1
):
    """
    Registers a polynomial decay learning rate scheduler.

    Args:
        optimizer (Any): The optimizer for which the learning rate scheduler is applied. Must be an instance of `torch.optim.Optimizer`.
        total_update_steps (int): The total number of training steps.
        max_decay_steps (int, optional): The maximum number of steps for decay. Defaults to -1, which means it will use `total_update_steps`.

    Returns:
        PolynomialDecayLR: An instance of the custom learning rate scheduler.
    """
    import torch
    from torch.optim.lr_scheduler import _LRScheduler

    class PolynomialDecayLR(_LRScheduler):
        """
        Implements a polynomial learning rate decay scheduler.

        The learning rate decreases following a polynomial function until the specified maximum decay step is reached.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to which the scheduler is applied.
            max_decay_steps (int): The step after which the learning rate stops decreasing.
            end_learning_rate (float, optional): The final learning rate value after decay. Defaults to 0.0001.
            power (float, optional): The power of the polynomial used for decay. Defaults to 1.0.
        """

        def __init__(
            self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0
        ):
            """
            Initializes the PolynomialDecayLR scheduler.

            Args:
                optimizer (torch.optim.Optimizer): The optimizer to which the scheduler is applied.
                max_decay_steps (int): The step after which the learning rate stops decreasing.
                end_learning_rate (float, optional): The final learning rate value after decay. Defaults to 0.0001.
                power (float, optional): The power of the polynomial used for decay. Defaults to 1.0.

            Raises:
                ValueError: If `max_decay_steps` is less than or equal to 1.
            """
            if max_decay_steps <= 1.0:
                raise ValueError("max_decay_steps should be greater than 1.")
            self.max_decay_steps = max_decay_steps
            self.end_learning_rate = end_learning_rate
            self.power = power
            super().__init__(optimizer)

        def get_lr(self):
            """
            Computes the current learning rate based on the polynomial decay formula.

            Returns:
                list[float]: A list of learning rates for each parameter group.
            """
            if self.last_epoch > self.max_decay_steps:
                return [self.end_learning_rate for _ in self.base_lrs]

            return [
                (base_lr - self.end_learning_rate)
                * ((1 - self.last_epoch / self.max_decay_steps) ** (self.power))
                + self.end_learning_rate
                for base_lr in self.base_lrs
            ]

        def _get_closed_form_lr(self):
            """
            Computes the closed-form solution for the learning rate based on the polynomial decay formula.

            Returns:
                list[float]: A list of learning rates for each parameter group.
            """
            return [
                (base_lr - self.end_learning_rate)
                * ((1 - self.last_epoch / self.max_decay_steps) ** (self.power))
                + self.end_learning_rate
                for base_lr in self.base_lrs
            ]

    assert isinstance(optimizer, torch.optim.Optimizer), (
        "optimizer must be a torch optimizer"
    )
    max_decay_steps = total_update_steps if max_decay_steps == -1 else max_decay_steps
    return PolynomialDecayLR(optimizer, max_decay_steps=max_decay_steps)
