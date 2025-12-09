from atria_registry import ModuleRegistry, RegistryGroup

from atria_metrics.core import MetricConfig


class MetricsRegistryGroup(RegistryGroup[MetricConfig]):
    """Registry group for Metrics."""

    pass


ModuleRegistry().add_registry_group(
    name="METRIC",
    registry_group=MetricsRegistryGroup(name="metric", package="atria_metrics"),
)
METRIC: MetricsRegistryGroup = ModuleRegistry().get_registry_group("METRIC")  # type: ignore
