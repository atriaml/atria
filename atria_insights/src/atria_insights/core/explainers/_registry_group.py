from atria_registry import ModuleRegistry, RegistryGroup

from atria_insights.explainers._base import ExplainerConfig


class ExplainersRegistryGroup(RegistryGroup[ExplainerConfig]):
    """Registry group for explainers."""

    pass


ModuleRegistry().add_registry_group(
    name="EXPLAINER",
    registry_group=ExplainersRegistryGroup(name="explainer", package="atria_ml"),
)
EXPLAINER: ExplainersRegistryGroup = ModuleRegistry().get_registry_group("EXPLAINER")  # type: ignore
