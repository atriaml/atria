from typing import Any

from atria_logger import get_logger
from atria_types import RepresentationMixin
from pydantic import BaseModel

logger = get_logger(__name__)


class GradientConfig(RepresentationMixin, BaseModel):
    enable_grad_clipping: bool = False
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1


class LoggingConfig(RepresentationMixin, BaseModel):
    logging_steps: int = 100
    refresh_rate: int = 10
    log_gpu_stats: bool = False
    profile_time: bool = False
    log_to_tb: bool = True


class EarlyStoppingConfig(RepresentationMixin, BaseModel):
    enabled: bool = False
    monitored_metric: str = "val/loss"
    min_delta: float = 0.0
    patience: int = 3
    cumulative_delta: bool = False
    mode: str = "min"


class ModelCheckpointConfig(RepresentationMixin, BaseModel):
    enabled: bool = True
    dir: str = "checkpoints"
    n_saved: int = 1
    n_best_saved: int = 1
    monitored_metric: str = "validation/running_avg_loss"
    mode: str = "min"
    name_prefix: str = ""
    save_weights_only: bool = False
    load_weights_only: bool = False
    every_n_steps: int | None = None
    every_n_epochs: int = 1
    resume_from_checkpoint: bool = True

    def model_post_init(self, context):
        if self.every_n_steps is not None and self.every_n_epochs is not None:
            raise RuntimeError(
                "model_checkpoint_config.every_n_steps and model_checkpoint_config.every_n_epochs are mutually exclusive"
            )

    @property
    def save_every_iters(self):
        if self.every_n_epochs is not None:
            return self.every_n_epochs
        else:
            return self.every_n_steps

    @property
    def save_per_epoch(self):
        if self.every_n_epochs is not None:
            return True
        else:
            return False


class ModelEmaConfig(RepresentationMixin, BaseModel):
    enabled: bool = False
    momentum: float = 0.0001
    momentum_warmup: float = 0.0
    warmup_iters: int = 0
    update_every: int = 1


class WarmupConfig(RepresentationMixin, BaseModel):
    warmup_ratio: float | None = None
    warmup_steps: int | None = None

    def model_post_init(self, context: Any) -> None:
        if self.warmup_ratio is not None:
            if self.warmup_ratio < 0 or self.warmup_ratio > 1:
                raise ValueError("warmup_ratio must lie in range [0,1]")
            elif self.warmup_ratio is not None and self.warmup_steps is not None:
                logger.info(
                    "Both warmup_ratio and warmup_steps given, warmup_steps will override"
                    " any effect of warmup_ratio during training"
                )
