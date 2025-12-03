"""
Warmup Configuration Module

This module defines the `WarmupConfig` class, which provides configuration options
for linear warmup during training. It includes options for specifying warmup ratio
and warmup steps.

Classes:
    - WarmupConfig: Configuration class for warmup settings.

Dependencies:
    - pydantic: For data validation and settings management.
    - atria_core.logger: For logging utilities.
    - rich.pretty: For pretty-printing objects.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from typing import Any

from atria_core.logger.logger import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)


class WarmupConfig(BaseModel):
    """
    Configuration class for linear warmup settings during training.

    Attributes:
        warmup_ratio (Optional[float]): Fraction of total steps for linear warmup. Defaults to None.
        warmup_steps (Optional[int]): Number of steps for linear warmup. Defaults to None.
    """

    warmup_ratio: float | None = None
    warmup_steps: int | None = None

    def model_post_init(self, context: Any) -> None:
        """
        Post-initialization method to validate and set warmup configuration.

        Args:
            context (Any): Additional context for initialization.

        Raises:
            ValueError: If `warmup_ratio` is not in the range [0, 1].
        """
        if self.warmup_ratio is not None:
            if self.warmup_ratio < 0 or self.warmup_ratio > 1:
                raise ValueError("warmup_ratio must lie in range [0,1]")
            elif self.warmup_ratio is not None and self.warmup_steps is not None:
                logger.info(
                    "Both warmup_ratio and warmup_steps given, warmup_steps will override"
                    " any effect of warmup_ratio during training"
                )

    def __repr__(self):
        """
        Returns a pretty-printed string representation of the warmup configuration.

        Returns:
            str: Pretty-printed string representation.
        """
        from rich.pretty import pretty_repr

        return pretty_repr(self)

    def __str__(self):
        """
        Returns a string representation of the warmup configuration.

        Returns:
            str: String representation.
        """
        return repr(self)
