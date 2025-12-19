from __future__ import annotations

import typing

from atria_registry import RegistryGroup
from atria_registry._module_registry import ModuleRegistry
from pydantic import Field

from atria_models.core.model_pipelines._common import T_ModelPipelineConfig
from atria_models.core.models.transformers._configs._encoder_model import (
    T_TransformersEncoderModelConfig,
)

T_ModelConfig = typing.Annotated[
    T_TransformersEncoderModelConfig, Field(discriminator="type")
]


class ModelPipelineRegistryGroup(RegistryGroup[T_ModelPipelineConfig]):
    def load_module_config(self, module_path: str, **kwargs) -> T_ModelPipelineConfig:
        """Dynamically load all registered modules in the registry group."""
        config = typing.cast(
            T_ModelPipelineConfig, super().load_module_config(module_path, **kwargs)
        )
        return config


class ModelRegistryGroup(RegistryGroup[T_ModelConfig]):
    def load_module_config(self, module_path: str, **kwargs) -> T_ModelConfig:
        """Dynamically load all registered modules in the registry group."""
        config = typing.cast(
            T_ModelConfig, super().load_module_config(module_path, **kwargs)
        )
        return config


ModuleRegistry().add_registry_group(
    name="MODEL_PIPELINES",
    registry_group=ModelPipelineRegistryGroup(
        name="model_pipelines", package="atria_models"
    ),
)
ModuleRegistry().add_registry_group(
    name="MODELS",
    registry_group=ModelRegistryGroup(name="models", package="atria_models"),
)
MODEL_PIPELINES: ModelPipelineRegistryGroup = typing.cast(
    ModelPipelineRegistryGroup, ModuleRegistry().MODEL_PIPELINES
)
MODELS: ModelRegistryGroup = typing.cast(ModelRegistryGroup, ModuleRegistry().MODELS)
