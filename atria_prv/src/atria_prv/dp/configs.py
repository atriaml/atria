from typing import Self

from atria_ml.configs._task import TrainingTaskConfig
from pydantic import BaseModel, model_validator


class DPConfig(BaseModel):
    noise_multiplier: float | None = None
    target_epsilon: float | None = None
    target_delta: float | None = None  # 1 / len (dataset)
    max_grad_norm: float = 1.0
    sample_rate: float | None = None
    loss_reduction: str = "mean"
    clipping: str = "flat"
    accountant: str = "rdp"
    max_physical_batch_size: int = 8
    use_bmm: bool = True

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.noise_multiplier is None and self.target_epsilon is None:
            raise RuntimeError(
                "At least one of `noise_multiplier` or `target_epsilon` must be provided."
            )
        return self


class DPTrainingTaskConfig(TrainingTaskConfig):
    dp_config: DPConfig
