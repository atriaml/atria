"""
Logging Configuration Module

This module defines the `LoggingConfig` class, which provides configuration options
for logging during training. It includes options for logging steps, refresh rates,
GPU stats, profiling, and TensorBoard logging.

Classes:
    - LoggingConfig: Configuration class for logging settings.

Dependencies:
    - pydantic: For data validation and settings management.
    - rich.pretty: For pretty-printing objects.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    """
    Configuration class for logging settings during training.

    Attributes:
        logging_steps (int): Number of update steps to log on. Defaults to 100.
        refresh_rate (int): Refresh rate for progress bar updates. Defaults to 10.
        log_gpu_stats (Optional[bool]): Whether to log GPU stats. Defaults to False.
        profile_time (Optional[bool]): Whether to enable profiling. Defaults to False.
        log_to_tb (Optional[bool]): Whether to log outputs to TensorBoard. Defaults to True.
    """

    logging_steps: int = 100
    refresh_rate: int = 10
    log_gpu_stats: bool = False
    profile_time: bool = False
    log_to_tb: bool = True

    def __repr__(self):
        """
        Returns a pretty-printed string representation of the logging configuration.

        Returns:
            str: Pretty-printed string representation.
        """
        from rich.pretty import pretty_repr

        return pretty_repr(self)

    def __str__(self):
        """
        Returns a string representation of the logging configuration.

        Returns:
            str: String representation.
        """
        return repr(self)
