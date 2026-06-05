from typing import Annotated, ClassVar, Literal

from pydantic import Field

from atria_ml.optimizers._base import OptimizerConfig
from atria_ml.optimizers._registry_group import OPTIMIZERS


@OPTIMIZERS.register("adam")
class AdamOptimizerConfig(OptimizerConfig):
    __module_path__: ClassVar[str] = "torch.optim.Adam"
    type: Literal["adam"] = "adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    amsgrad: bool = False


@OPTIMIZERS.register("adamw")
class AdamWOptimizerConfig(OptimizerConfig):
    __module_path__: ClassVar[str] = "torch.optim.AdamW"
    type: Literal["adamw"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    amsgrad: bool = False


@OPTIMIZERS.register("adagrad")
class AdagradOptimizerConfig(OptimizerConfig):
    __module_path__: ClassVar[str] = "torch.optim.Adagrad"
    type: Literal["adagrad"] = "adagrad"
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    eps: float = 1e-10


@OPTIMIZERS.register("rmsprop")
class RMSpropOptimizerConfig(OptimizerConfig):
    __module_path__: ClassVar[str] = "torch.optim.RMSprop"
    type: Literal["rmsprop"] = "rmsprop"
    alpha: float = 0.99
    eps: float = 1e-08
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False


@OPTIMIZERS.register("adadelta")
class AdadeltaOptimizerConfig(OptimizerConfig):
    __module_path__: ClassVar[str] = "torch.optim.Adadelta"
    type: Literal["adadelta"] = "adadelta"
    rho: float = 0.9
    eps: float = 1e-06
    weight_decay: float = 0.0


@OPTIMIZERS.register("sgd")
class SGDOptimizerConfig(OptimizerConfig):
    __module_path__: ClassVar[str] = "torch.optim.SGD"
    type: Literal["sgd"] = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False


@OPTIMIZERS.register("lars")
class LARSOptimizerConfig(OptimizerConfig):
    __module_path__: ClassVar[str] = "atria_ml.optimizers._lars.LARS"
    type: Literal["lars"] = "lars"
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    eta: float = 0.001


OptimizerConfigType = Annotated[
    AdamOptimizerConfig
    | AdamWOptimizerConfig
    | AdagradOptimizerConfig
    | RMSpropOptimizerConfig
    | AdadeltaOptimizerConfig
    | SGDOptimizerConfig
    | LARSOptimizerConfig,
    Field(discriminator="type"),
]
