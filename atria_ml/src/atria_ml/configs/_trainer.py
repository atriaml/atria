from __future__ import annotations

from atria_logger import get_logger
from atria_registry._module_base import BaseModel
from atria_types._utilities._repr import RepresentationMixin
from pydantic import ConfigDict, Field

from atria_ml.optimizers._configs import OptimizerConfigType, SGDOptimizerConfig
from atria_ml.schedulers._base import LRSchedulerConfig
from atria_ml.schedulers._configs import CosineAnnealingLRSchedulerConfig
from atria_ml.training._configs import (
    EarlyStoppingConfig,
    GradientConfig,
    ModelCheckpointConfig,
    ModelEmaConfig,
    WarmupConfig,
)

logger = get_logger(__name__)


class TrainerConfig(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    max_epochs: int = 10
    validate_every_n_epochs: float = 1.0
    visualize_every_n_epochs: float = 1.0
    resume_checkpoint_path: str | None = None
    clear_cuda_cache: bool = True
    stop_on_nan: bool = True
    eval_training: bool = False
    outputs_to_running_avg: list[str] = Field(default_factory=lambda: ["loss"])

    optimizer: dict[str, OptimizerConfigType] | OptimizerConfigType = (
        SGDOptimizerConfig(lr=0.1, momentum=0.9, weight_decay=0.0)
    )
    lr_scheduler: dict[str, LRSchedulerConfig] | LRSchedulerConfig | None = (
        CosineAnnealingLRSchedulerConfig()
    )
    model_ema: ModelEmaConfig = ModelEmaConfig()
    warmup: WarmupConfig = WarmupConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()
    gradient: GradientConfig = GradientConfig()
