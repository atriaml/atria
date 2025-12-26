import typing

from atria_registry import ModuleRegistry, RegistryGroup

from atria_ml.optimizers._base import OptimizerConfig


class OptimizersRegistryGroup(RegistryGroup[OptimizerConfig]):
    """Registry group for optimizers."""

    def load_module_config(self, module_path: str, **kwargs) -> OptimizerConfig:
        """Dynamically load all registered modules in the registry group."""
        config = super().load_module_config(module_path, **kwargs)
        assert isinstance(config, OptimizerConfig), (
            f"Loaded config is not an OptimizerConfig: {type(config)}"
        )
        return typing.cast(OptimizerConfig, config)


ModuleRegistry().add_registry_group(
    name="OPTIMIZERS",
    registry_group=OptimizersRegistryGroup(name="optimizers", package="atria_ml"),
)
OPTIMIZERS = ModuleRegistry().get_registry_group("OPTIMIZERS")
