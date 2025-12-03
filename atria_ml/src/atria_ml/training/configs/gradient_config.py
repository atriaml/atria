"""
Gradient Configuration Module

This module defines the `GradientConfig` class, which provides configuration options
for gradient-related settings during training. It includes options for gradient clipping,
maximum gradient norm, and gradient accumulation steps.

Classes:
    - GradientConfig: Configuration class for gradient settings.

Dependencies:
    - pydantic: For data validation and settings management.
    - rich.pretty: For pretty-printing objects.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from pydantic import BaseModel


class GradientConfig(BaseModel):
    """
    Configuration class for gradient-related settings during training.

    Attributes:
        enable_grad_clipping (bool): Whether to use gradient clipping. Defaults to False.
        max_grad_norm (float): Maximum gradient norm for clipping. Defaults to 1.0.
        gradient_accumulation_steps (int): Number of update steps to accumulate before performing a backward/update pass. Defaults to 1.
    """

    enable_grad_clipping: bool = False
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    def __repr__(self):
        """
        Returns a pretty-printed string representation of the gradient configuration.

        Returns:
            str: Pretty-printed string representation.
        """
        from rich.pretty import pretty_repr

        return pretty_repr(self)

    def __str__(self):
        """
        Returns a string representation of the gradient configuration.

        Returns:
            str: String representation.
        """
        return repr(self)
