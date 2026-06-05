import typing

from atria_registry import ModuleRegistry, RegistryGroup

from atria_metrics.core import MetricConfig


class MetricsRegistryGroup(RegistryGroup[MetricConfig]):
    """Registry group for Metrics."""

    def load_module_config(self, module_path: str, **kwargs) -> MetricConfig:
        """Dynamically load all registered modules in the registry group."""
        config = super().load_module_config(module_path, **kwargs)
        assert isinstance(config, MetricConfig), (
            f"Loaded config is not an MetricConfig: {type(config)}"
        )
        return typing.cast(MetricConfig, config)


ModuleRegistry().add_registry_group(
    name="METRICS",
    registry_group=MetricsRegistryGroup(name="metrics", package="atria_metrics"),
)
METRICS: MetricsRegistryGroup = ModuleRegistry().get_registry_group("METRICS")  # type: ignore

