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
    >>> # Get a registered model
    >>> model_cls = MODEL.get("my_model")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_registry import ModuleRegistry, RegistryGroup

if TYPE_CHECKING:
    from atria_datasets.core.dataset._common import DatasetConfig


class DatasetRegistryGroup(RegistryGroup["DatasetConfig"]):
    def load_module_config(self, module_path: str, **kwargs) -> DatasetConfig:
        """Dynamically load all registered modules in the registry group."""
        from atria_datasets.core.dataset._common import DatasetConfig

        config = super().load_module_config(module_path, **kwargs)
        assert isinstance(config, DatasetConfig)
        return config


ModuleRegistry().add_registry_group(
    name="DATASETS",
    registry_group=DatasetRegistryGroup(name="datasets", package="atria_datasets"),
)

DATASETS: DatasetRegistryGroup = ModuleRegistry().DATASETS
"""Registry group for datasets.

Used to register and manage dataset-related components throughout the application.
Provides methods to register new datasets and retrieve existing ones by name.
"""
