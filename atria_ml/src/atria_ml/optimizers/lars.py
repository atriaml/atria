"""
LARS Optimizer Module

This module defines the `LARS` optimizer, which implements Layer-wise Adaptive Rate Scaling (LARS)
for Stochastic Gradient Descent (SGD). It is designed for large-batch training of convolutional networks.

Classes:
    - LARS: Optimizer implementing the LARS algorithm.

Dependencies:
    - torch: For PyTorch operations.
    - atria_ml.registry: For registering the optimizer in the Atria framework.

References:
    - You, Gitman, and Ginsburg. "Large Batch Training of Convolutional Networks."
      https://arxiv.org/abs/1708.03888

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    r"""
    Implements Layer-wise Adaptive Rate Scaling (LARS) for SGD.

    Attributes:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Base learning rate (\gamma_0).
        momentum (float): Momentum factor (default: 0).
        weight_decay (float): Weight decay (L2 penalty) (default: 0).
        dampening (float): Dampening for momentum (default: 0).
        eta (float): LARS coefficient (default: 0.001).
        nesterov (bool): Enables Nesterov momentum (default: False).
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        eta=0.001,
        nesterov=False,
    ):
        """
        Initialize the LARS optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Base learning rate.
            momentum (float): Momentum factor.
            dampening (float): Dampening for momentum.
            weight_decay (float): Weight decay (L2 penalty).
            eta (float): LARS coefficient.
            nesterov (bool): Enables Nesterov momentum.

        Raises:
            ValueError: If any of the input parameters are invalid.
        """
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eta < 0.0:
            raise ValueError(f"Invalid LARS coefficient value: {eta}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "eta": eta,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        """
        Set the state of the optimizer.

        Args:
            state (dict): State dictionary.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            eta = group["eta"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            lars_exclude = group.get("lars_exclude", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                if lars_exclude:
                    local_lr = 1.0
                else:
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(d_p).item()
                    # Compute local learning rate for this layer
                    local_lr = (
                        eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                    )

                actual_lr = local_lr * lr
                d_p = d_p.add(p, alpha=weight_decay).mul(actual_lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-d_p)

        return loss
