"""
Model Checkpoint Configuration Module

This module defines the `ModelCheckpointConfig` class, which provides configuration options
for saving and loading model checkpoints during training. It includes options for checkpoint
directory, frequency, monitored metrics, and resume settings.

Classes:
    - ModelCheckpointConfig: Configuration class for model checkpoint settings.

Dependencies:
    - pydantic: For data validation and settings management.
    - rich.pretty: For pretty-printing objects.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from pydantic import BaseModel


class ModelCheckpointConfig(BaseModel):
    """
    Configuration class for saving and loading model checkpoints during training.

    Attributes:
        enabled (bool): Whether to enable checkpointing. Defaults to True.
        dir (str): Directory name for saving checkpoints. Defaults to "checkpoints".
        n_saved (int): Number of checkpoints to save. Defaults to 1.
        n_best_saved (int): Number of best checkpoints to save. Defaults to 1.
        monitored_metric (Optional[str]): Metric to monitor for saving best checkpoints. Defaults to "validation/running_avg_loss".
        mode (str): Mode for monitoring metric, either "min" or "max". Defaults to "min".
        name_prefix (str): Prefix for checkpoint file names. Defaults to an empty string.
        save_weights_only (bool): Whether to save only model weights. Defaults to False.
        load_weights_only (bool): Whether to load only model weights. Defaults to False.
        every_n_steps (Optional[int]): Save checkpoints every n steps. Defaults to None.
        every_n_epochs (Optional[int]): Save checkpoints every n epochs. Defaults to 1.
        resume_from_checkpoint (Optional[bool]): Whether to resume training from a checkpoint if available. Defaults to True.
    """

    enabled: bool = True
    dir: str = "checkpoints"
    n_saved: int = 1
    n_best_saved: int = 1
    monitored_metric: str = "validation/running_avg_loss"
    mode: str = "min"
    name_prefix: str = ""
    save_weights_only: bool = False
    load_weights_only: bool = False
    every_n_steps: int | None = None
    every_n_epochs: int = 1
    resume_from_checkpoint: bool = True

    def model_post_init(self, context):
        """
        Post-initialization method to validate checkpoint configuration.

        Args:
            context: Additional context for initialization.

        Raises:
            RuntimeError: If both `every_n_steps` and `every_n_epochs` are specified.
        """
        if self.every_n_steps is not None and self.every_n_epochs is not None:
            raise RuntimeError(
                "model_checkpoint_config.every_n_steps and model_checkpoint_config.every_n_epochs are mutually exclusive"
            )

    @property
    def save_every_iters(self):
        """
        Determine the frequency of saving checkpoints in iterations.

        Returns:
            int: Frequency of saving checkpoints in iterations.
        """
        if self.every_n_epochs is not None:
            return self.every_n_epochs
        else:
            return self.every_n_steps

    @property
    def save_per_epoch(self):
        """
        Check if checkpoints are saved per epoch.

        Returns:
            bool: True if checkpoints are saved per epoch, False otherwise.
        """
        if self.every_n_epochs is not None:
            return True
        else:
            return False

    def __repr__(self):
        """
        Returns a pretty-printed string representation of the checkpoint configuration.

        Returns:
            str: Pretty-printed string representation.
        """
        from rich.pretty import pretty_repr

        return pretty_repr(self)

    def __str__(self):
        """
        Returns a string representation of the checkpoint configuration.

        Returns:
            str: String representation.
        """
        return repr(self)
