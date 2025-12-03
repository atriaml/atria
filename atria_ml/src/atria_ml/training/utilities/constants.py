"""
Constants for Training Utilities

This module defines constants and utility classes for managing different stages of training and GAN-specific stages.

Classes:
    - TrainingStage: Defines constants for various training stages and provides a utility method to retrieve stage names.
    - GANStage: Defines constants for GAN-specific training stages.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from typing import Any


class TrainingStage:
    """
    Defines constants for various training stages.

    Attributes:
        train (str): Represents the training stage.
        validation (str): Represents the validation stage.
        test (str): Represents the testing stage.
        inference (str): Represents the inference stage.
        predict (str): Represents the prediction stage.
        visualization (str): Represents the visualization stage.

    Methods:
        get(name: str) -> Any: Retrieves the value of a training stage by its name.
    """

    train = "train"
    validation = "validation"
    test = "test"
    inference = "inference"
    predict = "predict"
    visualization = "visualization"

    @classmethod
    def get(cls, name: str) -> Any:
        """
        Retrieves the value of a training stage by its name.

        Args:
            name (str): The name of the training stage to retrieve.

        Returns:
            Any: The value of the specified training stage.

        Raises:
            AttributeError: If the specified stage name does not exist.
        """
        return getattr(cls, name)


class GANStage:
    """
    Defines constants for GAN-specific training stages.

    Attributes:
        train_generator (str): Represents the stage for training the generator.
        train_discriminator (str): Represents the stage for training the discriminator.
    """

    train_generator = "train_gen"
    train_discriminator = "train_disc"
