from atria_registry import ModuleRegistry, RegistryGroup

from atria_insights.core.explainability_metrics._base import ExplainabilityMetricConfig


class ExplainabilityMetricsRegistryGroup(RegistryGroup[ExplainabilityMetricConfig]):
    """Registry group for explainers."""

    pass


ModuleRegistry().add_registry_group(
    name="EXPLAINABILITY_METRIC",
    registry_group=ExplainabilityMetricsRegistryGroup(
        name="explainability_metric", package="atria_ml"
    ),
)
EXPLAINABILITY_METRIC: ExplainabilityMetricsRegistryGroup = (
    ModuleRegistry().get_registry_group("EXPLAINABILITY_METRIC")
)  # type: ignore
