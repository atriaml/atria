from atria_ml.optimizers._base import OptimizerConfig
from atria_ml.optimizers._registry_group import OPTIMIZER


@OPTIMIZER.register("adam")
class AdamOptimizerConfig(OptimizerConfig):
    module_path: str | None = "torch.optim.Adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    amsgrad: bool = False


@OPTIMIZER.register("adamw")
class AdamWOptimizerConfig(OptimizerConfig):
    module_path: str | None = "torch.optim.AdamW"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    amsgrad: bool = False


@OPTIMIZER.register("adagrad")
class AdagradOptimizerConfig(OptimizerConfig):
    module_path: str | None = "torch.optim.Adagrad"
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    eps: float = 1e-10


@OPTIMIZER.register("rmsprop")
class RMSpropOptimizerConfig(OptimizerConfig):
    module_path: str | None = "torch.optim.RMSprop"
    alpha: float = 0.99
    eps: float = 1e-08
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False


@OPTIMIZER.register("adadelta")
class AdadeltaOptimizerConfig(OptimizerConfig):
    module_path: str | None = "torch.optim.Adadelta"
    rho: float = 0.9
    eps: float = 1e-06
    weight_decay: float = 0.0


@OPTIMIZER.register("sgd")
class SGDOptimizerConfig(OptimizerConfig):
    module_path: str | None = "torch.optim.SGD"
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False


@OPTIMIZER.register("lars")
class LARSOptimizerConfig(OptimizerConfig):
    module_path: str | None = "atria_ml.optimizers._lars.LARS"
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    eta: float = 0.001
