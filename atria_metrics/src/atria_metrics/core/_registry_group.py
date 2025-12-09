from atria_registry import ModuleRegistry, RegistryGroup

from atria_metrics.core import MetricConfig


class MetricsRegistryGroup(RegistryGroup[MetricConfig]):
    """Registry group for Metrics."""

    pass


ModuleRegistry().add_registry_group(
    name="Metric",
    registry_group=MetricsRegistryGroup(name="metric", package="atria_metrics"),
)
METRIC = ModuleRegistry().get_registry_group("METRIC")
