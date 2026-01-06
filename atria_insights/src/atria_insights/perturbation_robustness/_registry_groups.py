from __future__ import annotations

import typing

from atria_registry import RegistryGroup
from atria_registry._module_registry import ModuleRegistry

from atria_insights.perturbation_robustness.pipelines._config import (
    T_PerturbationRobustnessPipelineConfig,
)


class PerturbationRobustnessPipelineRegistryGroup(
    RegistryGroup[T_PerturbationRobustnessPipelineConfig]
):
    def load_module_config(
        self, module_path: str, **kwargs
    ) -> T_PerturbationRobustnessPipelineConfig:
        """Dynamically load all registered modules in the registry group."""
        config = typing.cast(
            T_PerturbationRobustnessPipelineConfig,
            super().load_module_config(module_path, **kwargs),
        )
        return config


ModuleRegistry().add_registry_group(
    name="PERTURBATION_ROBUSTNESS_PIPELINES",
    registry_group=PerturbationRobustnessPipelineRegistryGroup(
        name="perturbation_robustness_pipelines", package="atria_insights"
    ),
)
PERTURBATION_ROBUSTNESS_PIPELINES: PerturbationRobustnessPipelineRegistryGroup = (
    typing.cast(
        PerturbationRobustnessPipelineRegistryGroup,
        ModuleRegistry().PERTURBATION_ROBUSTNESS_PIPELINES,
    )
)
