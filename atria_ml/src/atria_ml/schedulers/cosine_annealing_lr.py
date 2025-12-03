"""
This module provides a cosine annealing learning rate scheduler.

The `cosine_annealing_lr` function is designed to create a learning rate scheduler
that adjusts the learning rate following a cosine annealing schedule. This is useful
for training scenarios where periodic learning rate adjustments are beneficial.

Imports:
    - Any: A generic type from the typing module.
    - LR_SCHEDULER: A registry for learning rate schedulers from the atria_ml.registry module.

Functions:
    - cosine_annealing_lr: Creates a PyTorch CosineAnnealingLR scheduler with optional restarts.
"""

from typing import Any

from atria_ml.registry import LR_SCHEDULER


@LR_SCHEDULER.register("cosine_annealing_lr")
def cosine_annealing_lr(
    optimizer: Any,
    total_update_steps: int,
    total_warmup_steps: int,
    steps_per_epoch: int,
    restarts: bool = False,
    eta_min: int = 0,
    last_epoch: int = -1,
):
    """
    Create a cosine annealing learning rate scheduler.

    This function returns a PyTorch `CosineAnnealingLR` scheduler that adjusts the learning rate
    following a cosine annealing schedule. Optionally, it can support restarts for periodic
    adjustments.

    Args:
        optimizer (Any): The optimizer for which to schedule the learning rate. Must be an instance
            of `torch.optim.Optimizer`.
        total_update_steps (int): The total number of update steps for the training process.
        total_warmup_steps (int): The number of warmup steps before the cosine annealing schedule begins.
        steps_per_epoch (int): The number of steps in each epoch. Used when restarts are enabled.
        restarts (bool, optional): Whether to enable periodic restarts in the cosine annealing schedule.
            Defaults to False.
        eta_min (int, optional): The minimum learning rate value. Defaults to 0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR: A PyTorch learning rate scheduler instance.

    Raises:
        AssertionError: If the provided optimizer is not an instance of `torch.optim.Optimizer`.
    """
    import torch

    assert isinstance(optimizer, torch.optim.Optimizer), (
        "optimizer must be a torch optimizer"
    )
    if not restarts:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_update_steps - total_warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps_per_epoch - total_warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
