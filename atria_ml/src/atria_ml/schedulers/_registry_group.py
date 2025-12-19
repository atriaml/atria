from atria_registry import ModuleRegistry, RegistryGroup

from atria_ml.schedulers._base import LRSchedulerConfig


class LRSchedulerRegistryGroup(RegistryGroup[LRSchedulerConfig]):
    """Registry group for learning rate schedulers."""

    pass


ModuleRegistry().add_registry_group(
    name="LR_SCHEDULERS",
    registry_group=LRSchedulerRegistryGroup(name="lr_schedulers", package="atria_ml"),
)
LR_SCHEDULERS = ModuleRegistry().get_registry_group("LR_SCHEDULERS")
