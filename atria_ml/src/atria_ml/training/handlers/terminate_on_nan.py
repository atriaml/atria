"""
Terminate on NaN Handler Module

This module defines the `TerminateOnNan` class, which extends the functionality of the
`ignite.handlers.TerminateOnNan` to include additional checks for NaN or infinite values
in the output of the training process.

Classes:
    - TerminateOnNan: Custom handler for terminating training when NaN or Inf values are detected.

Dependencies:
    - torch: For PyTorch operations.
    - ignite: For engine and handler management.
    - atria_core.utilities.tensors: For tensor utilities.
    - atria_core.types.base.data_model: For data model structures.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

import numbers

import torch
from atria_core.types.base.data_model import BaseDataModel
from atria_core.utilities.tensors import _apply_to_type
from ignite.engine import Engine
from ignite.handlers import TerminateOnNan as IgniteTerminateOnNan


class TerminateOnNan(IgniteTerminateOnNan):
    """
    Custom handler for terminating training when NaN or Inf values are detected.

    This handler extends the `ignite.handlers.TerminateOnNan` to provide additional checks
    for NaN or infinite values in the output of the training process.
    """

    def __call__(self, engine: Engine) -> None:
        """
        Check the output of the engine for NaN or Inf values and terminate training if any are found.

        Args:
            engine (Engine): The engine instance to monitor.

        Raises:
            RuntimeError: If NaN or Inf values are detected in the output.
        """
        output = self._output_transform(engine.state.output)

        def raise_error(x: float | torch.Tensor) -> None:
            if x is None or isinstance(x, (BaseDataModel)):
                return

            if isinstance(x, numbers.Number):
                x = torch.tensor(x)

            if isinstance(x, torch.Tensor) and not bool(torch.isfinite(x).all()):
                raise RuntimeError("Infinite or NaN tensor found.")

        try:
            _apply_to_type(
                output,
                (numbers.Number, torch.Tensor, BaseDataModel),
                raise_error,
            )
        except RuntimeError:
            self.logger.warning(
                f"{self.__class__.__name__}: Output '{output}' contains NaN or Inf. Stop training"
            )
            engine.terminate()
