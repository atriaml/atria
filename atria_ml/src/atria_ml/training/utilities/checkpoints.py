"""
Checkpoint Utilities Module

This module provides utility functions for managing and locating checkpoint files during training and testing.
It includes functions to find specific checkpoint files based on user input or criteria such as the latest or best checkpoint.

Functions:
    - find_checkpoint_file: Locates a checkpoint file based on user input or criteria.
    - find_resume_checkpoint: Finds a checkpoint file for resuming training.
    - find_test_checkpoint: Finds a checkpoint file for testing a model.

Dependencies:
    - glob: For pattern matching file paths.
    - os: For file system operations.
    - pathlib.Path: For handling file paths.
    - atria_core.logger.logger: For logging messages.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from atria_core.logger.logger import get_logger

logger = get_logger(__name__)


def find_checkpoint_file_in_dir(checkpoint_dir: str, load_best: bool = False):
    """
    Locates a checkpoint file based on user input or criteria such as the latest or best checkpoint.

    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.
        load_best (bool, optional): Whether to prioritize finding the best checkpoint. Defaults to False.
        quiet (bool, optional): Whether to suppress logging messages. Defaults to False.

    Returns:
        pathlib.Path or None: The path to the located checkpoint file, or None if no file is found.

    Notes:
        - If `filename` is provided and exists, it is returned directly.
        - If no `filename` is provided, the function searches for checkpoint files in `checkpoint_dir`.
        - If `load_best` is True, only checkpoints containing "best" in their name are considered.
    """
    import glob
    import os

    if not checkpoint_dir.exists():
        return

    list_checkpoints = glob.glob(str(checkpoint_dir) + "/*.pt")
    if len(list_checkpoints) > 0:
        if not load_best:
            list_checkpoints = [c for c in list_checkpoints if "best" not in c]
        else:
            list_checkpoints = [c for c in list_checkpoints if "best" in c]

        if len(list_checkpoints) > 0:
            return max(list_checkpoints, key=os.path.getctime)


def find_resume_checkpoint(checkpoint_dir: str):
    """
    Finds a checkpoint file for resuming training.

    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.
        load_best (bool, optional): Whether to prioritize finding the best checkpoint. Defaults to False.

    Returns:
        pathlib.Path or None: The path to the located checkpoint file, or None if no file is found.

    Notes:
        - This function is a wrapper around `find_checkpoint_file` with `resume` set to True.
    """
    return find_checkpoint_file_in_dir(checkpoint_dir=checkpoint_dir, load_best=False)


def find_test_checkpoint(checkpoint_dir: str, load_best: bool = False):
    """
    Finds a checkpoint file for testing a model.

    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.
        load_best (bool, optional): Whether to prioritize finding the best checkpoint. Defaults to False.

    Returns:
        pathlib.Path or None: The path to the located checkpoint file, or None if no file is found.

    Notes:
        - This function is a wrapper around `find_checkpoint_file` with `resume` set to False.
    """
    return find_checkpoint_file_in_dir(
        checkpoint_dir=checkpoint_dir, load_best=load_best, resume=False
    )
