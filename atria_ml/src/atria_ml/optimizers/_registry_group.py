from atria_registry import ModuleRegistry, RegistryGroup

from atria_ml.optimizers._base import OptimizerConfig


class OptimizersRegistryGroup(RegistryGroup[OptimizerConfig]):
    """Registry group for optimizers."""

    pass


ModuleRegistry().add_registry_group(
    name="OPTIMIZER",
    registry_group=OptimizersRegistryGroup(name="optimizer", package="atria_ml"),
)
OPTIMIZER = ModuleRegistry().get_registry_group("OPTIMIZER")
