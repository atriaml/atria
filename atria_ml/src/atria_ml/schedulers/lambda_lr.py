"""
This module provides a LambdaLR learning rate scheduler.

The `lambda_lr` function is designed to create a PyTorch `LambdaLR` scheduler
that adjusts the learning rate based on a user-defined lambda function. This is
useful for implementing custom learning rate schedules, such as linear warmup
followed by linear decay.

Imports:
    - torch: PyTorch library for deep learning.
    - LR_SCHEDULER: A registry for learning rate schedulers from the `core.registry` module.

Functions:
    - lambda_lr: Creates a PyTorch `LambdaLR` scheduler with a linear warmup and decay schedule.
"""

from typing import Any

from atria_ml.registry import LR_SCHEDULER


@LR_SCHEDULER.register("lambda_lr")
def lambda_lr(
    optimizer: Any,
    total_update_steps: int,
    total_warmup_steps: int,
    lambda_fn: str = "linear",
    last_epoch: int = -1,
):
    """
    Create a LambdaLR learning rate scheduler with a custom lambda function.

    This function returns a PyTorch `LambdaLR` scheduler that adjusts the learning rate
    based on a user-defined lambda function. By default, it implements a linear warmup
    followed by a linear decay schedule.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        total_update_steps (int): The total number of training steps.
        total_warmup_steps (int): The number of warmup steps before the learning rate starts decaying.
        lambda_fn (str, optional): The name of the lambda function to use. Currently, only "linear"
            is supported. Defaults to "linear".
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: A PyTorch learning rate scheduler instance.

    Raises:
        ValueError: If an unknown `lambda_fn` is provided.
    """
    import torch

    assert isinstance(optimizer, torch.optim.Optimizer), (
        "optimizer must be a torch optimizer"
    )

    if lambda_fn == "linear":

        def linear_lambda_lr(current_step: int):
            """
            Compute the learning rate multiplier for the current step using a linear schedule.

            Args:
                current_step (int): The current training step.

            Returns:
                float: The learning rate multiplier for the current step.
            """
            if current_step < total_warmup_steps:
                return float(current_step) / float(max(1, total_warmup_steps))
            return max(
                0.0,
                float(total_update_steps - current_step)
                / float(max(1, total_update_steps - total_warmup_steps)),
            )

        lambda_fn = linear_lambda_lr
    else:
        raise ValueError(f"Unknown lambda_fn: {lambda_fn}")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda_fn, last_epoch=last_epoch
    )
