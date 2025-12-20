from __future__ import annotations

import typing

from atria_registry import RegistryGroup
from atria_registry._module_registry import ModuleRegistry

from atria_insights.core.model_pipelines._common import T_ExplainableModelPipelineConfig


class ExplainableModelPipelineRegistryGroup(
    RegistryGroup[T_ExplainableModelPipelineConfig]
):
    def load_module_config(
        self, module_path: str, **kwargs
    ) -> T_ExplainableModelPipelineConfig:
        """Dynamically load all registered modules in the registry group."""
        config = typing.cast(
            T_ExplainableModelPipelineConfig,
            super().load_module_config(module_path, **kwargs),
        )
        return config


ModuleRegistry().add_registry_group(
    name="EXPLAINABLE_MODEL_PIPELINES",
    registry_group=ExplainableModelPipelineRegistryGroup(
        name="explainable_model_pipelines", package="atria_insights"
    ),
)
EXPLAINABLE_MODEL_PIPELINES: ExplainableModelPipelineRegistryGroup = typing.cast(
    ExplainableModelPipelineRegistryGroup, ModuleRegistry().EXPLAINABLE_MODEL_PIPELINES
)
