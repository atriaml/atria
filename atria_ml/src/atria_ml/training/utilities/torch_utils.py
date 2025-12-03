"""
Torch Utilities Module

This module provides utility functions for working with PyTorch models and devices.
It includes functionality for moving modules to specific devices, preparing
modules for distributed training, and setting up TensorBoard logging.

Functions:
    - _reset_random_seeds: Resets random seeds for reproducibility across various libraries.
    - _initialize_torch: Initializes PyTorch settings, including random seeds and deterministic behavior.
    - _setup_tensorboard: Sets up TensorBoard logging for distributed training.

Dependencies:
    - torch: For PyTorch operations.
    - ignite.distributed: For distributed training utilities.
    - atria_ml.training.utilities.ddp_model_proxy.ModuleProxyWrapper: For wrapping distributed modules.
    - ignite.handlers.TensorboardLogger: For TensorBoard logging.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from pathlib import Path
from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger

if TYPE_CHECKING:
    from ignite.contrib.handlers.base_logger import BaseLogger

logger = get_logger(__name__)


def _reset_random_seeds(seed):
    """
    Resets random seeds for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set for random number generation.

    Libraries affected:
        - random: Python's built-in random module.
        - numpy: NumPy library for numerical computations.
        - torch: PyTorch library for deep learning.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _initialize_torch(seed: int = 0, deterministic: bool = False):
    """
    Initializes PyTorch settings, including random seeds and deterministic behavior.

    Args:
        seed (int, optional): The base seed value for random number generation. Defaults to 0.
        deterministic (bool, optional): Whether to enforce deterministic behavior for reproducibility. Defaults to False.

    Behavior:
        - Sets the global seed for reproducibility.
        - Configures PyTorch's CuDNN backend for deterministic or performance-optimized behavior.
    """
    import os

    import ignite.distributed as idist
    import torch

    seed = seed + idist.get_rank()
    _reset_random_seeds(seed)

    # Set seed as an environment variable
    os.environ["DEFAULT_SEED"] = str(seed)

    # Configure CuDNN backend for deterministic behavior if required
    if deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return seed


def _setup_tensorboard(output_dir: str) -> "BaseLogger":
    """
    Sets up TensorBoard logging for distributed training.

    Args:
        output_dir (str): The directory where TensorBoard logs will be saved.

    Returns:
        BaseLogger: An instance of Ignite's TensorboardLogger if the current process is the main process; otherwise, None.

    Notes:
        - Only the main process (rank 0) sets up TensorBoard logging.
        - The log directory is created under the specified `output_dir`.
    """
    import ignite.distributed as idist

    if idist.get_rank() == 0:
        from ignite.handlers import TensorboardLogger

        log_dir = Path(output_dir) / "tensorboard_log_dir"
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorboardLogger(log_dir=log_dir)
    else:
        tb_logger = None
    return tb_logger
