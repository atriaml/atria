import typing

from atria_registry import ModuleRegistry, RegistryGroup

from atria_insights.explainers._base import ExplainerConfig


class ExplainersRegistryGroup(RegistryGroup[ExplainerConfig]):
    """Registry group for explainers."""

    def load_module_config(self, module_path: str, **kwargs) -> ExplainerConfig:
        """Dynamically load all registered modules in the registry group."""
        config = super().load_module_config(module_path, **kwargs)
        assert isinstance(config, ExplainerConfig), (
            f"Loaded config is not an ExplainerConfig: {type(config)}"
        )
        return typing.cast(ExplainerConfig, config)


ModuleRegistry().add_registry_group(
    name="EXPLAINER",
    registry_group=ExplainersRegistryGroup(name="explainer", package="atria_insights"),
)
EXPLAINERS: ExplainersRegistryGroup = ModuleRegistry().get_registry_group("EXPLAINER")  # type: ignore
