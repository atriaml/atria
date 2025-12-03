from atria_registry import ModuleBuilder, ModuleRegistry, RegistryGroup


class OptimizerBuilder(ModuleBuilder):
    pass


class LRSchedulerBuilder(ModuleBuilder):
    pass


class OptimizersRegistryGroup(RegistryGroup):
    __registers_as_module_builder__ = True
    __module_builder_class__ = OptimizerBuilder
    __exclude_from_builder__ = [
        "params"  # these are passed at runtime from trainer
    ]


class LRSchedulerRegistryGroup(RegistryGroup):
    __registers_as_module_builder__ = True
    __module_builder_class__ = LRSchedulerBuilder
    __exclude_from_builder__ = [  # these are passed at runtime from trainer
        "optimizer",
        "total_warmup_steps",
        "total_update_steps",
        "total_warmup_steps",
        "steps_per_epoch",
    ]


ModuleRegistry().add_registry_group(
    name="TASK_PIPELINE",
    registry_group=RegistryGroup(name="task_pipeline", default_provider="atria_ml"),
)
ModuleRegistry().add_registry_group(
    name="LR_SCHEDULER",
    registry_group=LRSchedulerRegistryGroup(
        name="lr_scheduler", default_provider="atria_ml"
    ),
)
ModuleRegistry().add_registry_group(
    name="OPTIMIZER",
    registry_group=OptimizersRegistryGroup(
        name="optimizer", default_provider="atria_ml"
    ),
)
ModuleRegistry().add_registry_group(
    name="ENGINE",
    registry_group=RegistryGroup(name="engine", default_provider="atria_ml"),
)

TASK_PIPELINE = ModuleRegistry().get_registry_group("TASK_PIPELINE")
LR_SCHEDULER = ModuleRegistry().get_registry_group("LR_SCHEDULER")
OPTIMIZER = ModuleRegistry().get_registry_group("OPTIMIZER")
ENGINE = ModuleRegistry().get_registry_group("ENGINE")
