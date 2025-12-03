"""
This module registers external learning rate scheduler modules to the LR_SCHEDULER registry.

The `ext_modules` module is responsible for adding PyTorch and Ignite learning rate schedulers
into the `LR_SCHEDULER` registry. This allows these schedulers to be easily accessed and used
throughout the training pipeline.

Imports:
    - LR_SCHEDULER: A registry for learning rate schedulers from the `atria_ml.registry` module.

Functionality:
    - Registers PyTorch learning rate schedulers such as StepLR, MultiStepLR, ExponentialLR, and CyclicLR.
    - Registers the Ignite ReduceLROnPlateauScheduler for advanced learning rate scheduling.
"""

from atria_ml.registry import LR_SCHEDULER

# Register PyTorch learning rate schedulers
for scheduler_name, scheduler_path in [
    ("step_lr", "torch.optim.lr_scheduler.StepLR"),
    ("multi_step_lr", "torch.optim.lr_scheduler.MultiStepLR"),
    ("exponential_lr", "torch.optim.lr_scheduler.ExponentialLR"),
    ("cyclic_lr", "torch.optim.lr_scheduler.CyclicLR"),
    ("reduce_lr_on_plateau", "ignite.handlers.ReduceLROnPlateauScheduler"),
]:
    LR_SCHEDULER.register(scheduler_name)(scheduler_path)
