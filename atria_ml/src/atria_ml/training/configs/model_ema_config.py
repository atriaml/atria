"""
Model EMA Configuration Module

This module defines the `ModelEmaConfig` class, which provides configuration options
for Exponential Moving Average (EMA) of model parameters during training. It includes
options for enabling EMA, setting momentum, warmup iterations, and update frequency.

Classes:
    - ModelEmaConfig: Configuration class for model EMA settings.

Dependencies:
    - pydantic: For data validation and settings management.
    - rich.pretty: For pretty-printing objects.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from pydantic import BaseModel


class ModelEmaConfig(BaseModel):
    """
    Configuration class for Exponential Moving Average (EMA) of model parameters.

    Attributes:
        enabled (bool): Whether to enable EMA. Defaults to False.
        momentum (float): EMA decay rate. Defaults to 0.0001.
        momentum_warmup (float): Warmup value for EMA momentum. Defaults to 0.0.
        warmup_iters (int): Number of warmup iterations. Defaults to 0.
        update_every (int): Frequency of EMA updates in epochs. Defaults to 1.
    """

    enabled: bool = False
    momentum: float = 0.0001
    momentum_warmup: float = 0.0
    warmup_iters: int = 0
    update_every: int = 1

    def __repr__(self):
        """
        Returns a pretty-printed string representation of the EMA configuration.

        Returns:
            str: Pretty-printed string representation.
        """
        from rich.pretty import pretty_repr

        return pretty_repr(self)

    def __str__(self):
        """
        Returns a string representation of the EMA configuration.

        Returns:
            str: String representation.
        """
        return repr(self)
