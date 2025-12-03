"""
Early Stopping Configuration Module

This module defines the `EarlyStoppingConfig` class, which provides configuration options
for early stopping during training. It includes options for enabling early stopping,
monitoring metrics, setting patience, and defining the stopping mode.

Classes:
    - EarlyStoppingConfig: Configuration class for early stopping settings.

Dependencies:
    - pydantic: For data validation and settings management.
    - rich.pretty: For pretty-printing objects.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from pydantic import BaseModel


class EarlyStoppingConfig(BaseModel):
    """
    Configuration class for early stopping settings during training.

    Attributes:
        enabled (bool): Whether to enable early stopping. Defaults to False.
        monitored_metric (Optional[str]): Metric to monitor for early stopping. Defaults to "val/loss".
        min_delta (float): Minimum delta between subsequent steps. Defaults to 0.0.
        patience (int): Number of steps to wait before stopping. Defaults to 3.
        cumulative_delta (bool): Whether to accumulate delta over steps. Defaults to False.
        mode (str): Mode to use for monitoring, either "min" or "max". Defaults to "min".
    """

    enabled: bool = False
    monitored_metric: str = "val/loss"
    min_delta: float = 0.0
    patience: int = 3
    cumulative_delta: bool = False
    mode: str = "min"

    def __repr__(self):
        """
        Returns a pretty-printed string representation of the early stopping configuration.

        Returns:
            str: Pretty-printed string representation.
        """
        from rich.pretty import pretty_repr

        return pretty_repr(self)

    def __str__(self):
        """
        Returns a string representation of the early stopping configuration.

        Returns:
            str: String representation.
        """
        return repr(self)
