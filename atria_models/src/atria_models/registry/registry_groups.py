from __future__ import annotations

from atria_registry import RegistryGroup
from atria_registry._module_registry import ModuleRegistry

from atria_models.core.model_pipelines._model_pipeline import ModelPipeline


class ModelPipelineRegistryGroup(RegistryGroup[ModelPipeline]):
    pass


ModuleRegistry().add_registry_group(
    name="MODEL_PIPELINE",
    registry_group=ModelPipelineRegistryGroup(
        name="model_pipeline", package="atria_models"
    ),
)
MODEL_PIPELINE: ModelPipelineRegistryGroup = ModuleRegistry().MODEL_PIPELINE
