from __future__ import annotations

import typing

from atria_registry import RegistryGroup
from atria_registry._module_registry import ModuleRegistry

from atria_insights.explanation_pipelines._common import T_ExplanationPipelineConfig


class ExplanationPipelineRegistryGroup(RegistryGroup[T_ExplanationPipelineConfig]):
    def load_module_config(
        self, module_path: str, **kwargs
    ) -> T_ExplanationPipelineConfig:
        """Dynamically load all registered modules in the registry group."""
        from atria_insights.explanation_pipelines._common import (
            T_ExplanationPipelineConfig,
        )

        config = typing.cast(
            T_ExplanationPipelineConfig,
            super().load_module_config(module_path, **kwargs),
        )
        return config


ModuleRegistry().add_registry_group(
    name="EXPLANATION_PIPELINES",
    registry_group=ExplanationPipelineRegistryGroup(
        name="explanation_pipeline", package="atria_insights"
    ),
)
EXPLANATION_PIPELINES: ExplanationPipelineRegistryGroup = typing.cast(
    ExplanationPipelineRegistryGroup, ModuleRegistry().EXPLANATION_PIPELINES
)
