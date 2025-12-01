# """
# Registry Initialization Module

# This module initializes the registry system for the Atria application. It imports
# and initializes various registry groups from the `ModuleRegistry` class, making
# them accessible as module-level constants. These registry groups are used to
# manage datasets, data pipelines, data transformations, and other components
# within the application.

# The registry system provides a centralized way to register and retrieve components
# such as datasets, models, transformations, and pipelines throughout the application.

# Constants:
#     DATA_TRANSFORM: Registry group for data transformation components
# Example:
#     >>> from atria_registry import DATA_TRANSFORM, MODEL
#     >>> # Register a new data transform
#     >>> @DATA_TRANSFORM.register()
#     >>> class MyTransform:
#     ...     pass
#     >>> # Get a registered model
#     >>> model_cls = MODEL.get("my_model")

# Dependencies:
#     atria_registry.module_registry: Provides the ModuleRegistry class
#     atria_registry.registry_group: Provides registry group classes

# Author: Atria Development Team
# Date: 2025-07-10
# Version: 1.2.0
# License: MIT
# """

# from atria_registry import ModuleRegistry

# from atria_transforms.registry.module_registry import init_registry
# from atria_transforms.registry.registry_groups import DataTransformRegistryGroup

# init_registry()

# DATA_TRANSFORM: DataTransformRegistryGroup = ModuleRegistry().DATA_TRANSFORM
# """Registry group for data transformations.

# Used to register and manage data transformation components that modify or process
# input data. Includes preprocessing, augmentation, and normalization operations.
# """

# __all__ = ["DATA_TRANSFORM"]
