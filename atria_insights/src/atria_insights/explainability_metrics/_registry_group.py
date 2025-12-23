from __future__ import annotations

import typing

from atria_registry import ModuleRegistry, RegistryGroup

from atria_insights.explainability_metrics._base import (
    ExplainabilityMetricConfig,  # noqa
)


class ExplainabilityMetricsRegistryGroup(RegistryGroup[ExplainabilityMetricConfig]):
    """Registry group for explainers."""

    def load_module_config(
        self, module_path: str, **kwargs
    ) -> ExplainabilityMetricConfig:
        """Dynamically load all registered modules in the registry group."""
        config = super().load_module_config(module_path, **kwargs)
        assert isinstance(config, ExplainabilityMetricConfig), (
            f"Loaded config is not an ExplainerConfig: {type(config)}"
        )
        return typing.cast(ExplainabilityMetricConfig, config)


ModuleRegistry().add_registry_group(
    name="EXPLAINABILITY_METRIC",
    registry_group=ExplainabilityMetricsRegistryGroup(
        name="explainability_metric", package="atria_insights"
    ),
)
EXPLAINABILITY_METRICS: ExplainabilityMetricsRegistryGroup = (
    ModuleRegistry().get_registry_group("EXPLAINABILITY_METRIC")
)  # type: ignore
