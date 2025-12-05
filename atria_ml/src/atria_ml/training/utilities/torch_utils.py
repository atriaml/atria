from typing import TYPE_CHECKING

from atria_logger import get_logger

if TYPE_CHECKING:
    pass

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
