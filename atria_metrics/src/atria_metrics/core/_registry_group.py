from atria_registry import ModuleRegistry, RegistryGroup

from atria_metrics.core import MetricConfig


class MetricsRegistryGroup(RegistryGroup[MetricConfig]):
    """Registry group for Metrics."""

    pass


ModuleRegistry().add_registry_group(
    name="METRICS",
    registry_group=MetricsRegistryGroup(name="metrics", package="atria_metrics"),
)
METRICS: MetricsRegistryGroup = ModuleRegistry().get_registry_group("METRICS")  # type: ignore
