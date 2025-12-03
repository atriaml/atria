"""
Optimizer Events Module

This module defines custom events for optimizers in the Atria framework. These events
can be used to trigger specific actions during the training process.

Classes:
    - OptimizerEvents: Enum class for defining optimizer-related events.

Dependencies:
    - ignite.engine: For defining custom events using EventEnum.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from ignite.engine import EventEnum


class OptimizerEvents(EventEnum):
    """
    Enum class for defining optimizer-related events.

    Attributes:
        optimizer_step (str): Event triggered at each optimizer step.
    """

    optimizer_step = "optimizer_step"
