from __future__ import annotations

import typing

from atria_registry import RegistryGroup
from atria_registry._module_registry import ModuleRegistry

from atria_insights.feature_perturbation.evaluator_pipelines._config import (
    T_FeaturePerturbationEvaluatorPipelineConfig,
)


class FeaturePerturbationEvaluatorPipelineRegistryGroup(
    RegistryGroup[T_FeaturePerturbationEvaluatorPipelineConfig]
):
    def load_module_config(
        self, module_path: str, **kwargs
    ) -> T_FeaturePerturbationEvaluatorPipelineConfig:
        """Dynamically load all registered modules in the registry group."""
        config = typing.cast(
            T_FeaturePerturbationEvaluatorPipelineConfig,
            super().load_module_config(module_path, **kwargs),
        )
        return config


ModuleRegistry().add_registry_group(
    name="FEATURE_PERTURBATION_EVALUATOR_PIPELINES",
    registry_group=FeaturePerturbationEvaluatorPipelineRegistryGroup(
        name="feature_perturbation_evaluator_pipelines", package="atria_insights"
    ),
)
FEATURE_PERTURBATION_EVALUATOR_PIPELINES: FeaturePerturbationEvaluatorPipelineRegistryGroup = typing.cast(
    FeaturePerturbationEvaluatorPipelineRegistryGroup,
    ModuleRegistry().FEATURE_PERTURBATION_EVALUATOR_PIPELINES,
)
