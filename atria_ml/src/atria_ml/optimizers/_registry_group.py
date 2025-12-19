from atria_registry import ModuleRegistry, RegistryGroup

from atria_ml.optimizers._base import OptimizerConfig


class OptimizersRegistryGroup(RegistryGroup[OptimizerConfig]):
    """Registry group for optimizers."""

    pass


ModuleRegistry().add_registry_group(
    name="OPTIMIZERS",
    registry_group=OptimizersRegistryGroup(name="optimizers", package="atria_ml"),
)
OPTIMIZERS = ModuleRegistry().get_registry_group("OPTIMIZERS")
