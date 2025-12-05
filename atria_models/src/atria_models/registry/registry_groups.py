from __future__ import annotations

import typing

from atria_registry import RegistryGroup
from atria_registry._module_registry import ModuleRegistry

from atria_models.core.model_pipelines._common import T_ModelPipelineConfig


class ModelPipelineRegistryGroup(RegistryGroup[T_ModelPipelineConfig]):
    def load_module_config(self, module_path: str, **kwargs) -> T_ModelPipelineConfig:
        """Dynamically load all registered modules in the registry group."""
        config = typing.cast(
            T_ModelPipelineConfig, super().load_module_config(module_path, **kwargs)
        )
        return config


ModuleRegistry().add_registry_group(
    name="MODEL_PIPELINE",
    registry_group=ModelPipelineRegistryGroup(
        name="model_pipeline", package="atria_models"
    ),
)
MODEL_PIPELINE: ModelPipelineRegistryGroup = ModuleRegistry().MODEL_PIPELINE
