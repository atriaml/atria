from atria_registry import ModuleRegistry, RegistryGroup

from atria_ml.optimizers._base import OptimizerConfig
from atria_ml.schedulers._base import LRSchedulerConfig


class OptimizersRegistryGroup(RegistryGroup[OptimizerConfig]):
    """Registry group for optimizers."""

    pass


class LRSchedulerRegistryGroup(RegistryGroup[LRSchedulerConfig]):
    """Registry group for learning rate schedulers."""

    pass


ModuleRegistry().add_registry_group(
    name="OPTIMIZER",
    registry_group=OptimizersRegistryGroup(name="optimizer", package="atria_ml"),
)
ModuleRegistry().add_registry_group(
    name="LR_SCHEDULER",
    registry_group=LRSchedulerRegistryGroup(name="lr_scheduler", package="atria_ml"),
)
OPTIMIZER = ModuleRegistry().get_registry_group("OPTIMIZER")
LR_SCHEDULER = ModuleRegistry().get_registry_group("LR_SCHEDULER")
