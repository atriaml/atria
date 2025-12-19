"""Datasets registry initialization module

The registry system provides a centralized way to register and retrieve components
such as datasets, models, transformations, and pipelines throughout the application.

Attributes:
    DATASET: Registry group for dataset components
    DATA_PIPELINE: Registry group for data pipeline components
    BATCH_SAMPLER: Registry group for batch sampler components

Example:
    >>> from atria_registry import DATA_TRANSFORM, MODEL
    >>> # Register a new data transform
    >>> @DATA_TRANSFORM.register()
    >>> class MyTransform:
    ...     pass
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_registry import ModuleRegistry, RegistryGroup

if TYPE_CHECKING:
    from atria_transforms.core._tfs._base import DataTransform  # noqa


class DataTransformRegistryGroup(RegistryGroup["DataTransform"]):
    pass


ModuleRegistry().add_registry_group(
    name="DATA_TRANSFORMS",
    registry_group=DataTransformRegistryGroup(
        name="data_transforms", package="atria_transforms"
    ),
)

DATA_TRANSFORMS: DataTransformRegistryGroup = ModuleRegistry().DATA_TRANSFORMS
"""Registry group for data transformations."""
