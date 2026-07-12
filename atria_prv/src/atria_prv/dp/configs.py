from atria_ml.configs._task import TrainingTaskConfig
from pydantic import BaseModel


class DPConfig(BaseModel):
    noise_multiplier: float | None = None
    target_epsilon: float = 5.0
    target_delta: float | None = None  # 1 / len (dataset)
    max_grad_norm: float = 1.0
    sample_rate: float | None = None
    loss_reduction: str = "mean"
    clipping: str = "flat"
    accountant: str = "rdp"
    global_clip: float | None = None
    max_physical_batch_size: int = 8
    use_bmm: bool = True


class DPTrainingTaskConfig(TrainingTaskConfig):
    dp_config: DPConfig


# class ServerAdamConfig(BaseModel):
#     lr: float = Field(
#         default=1e-2, description="Server-side global Adam learning rate (eta)."
#     )
#     beta1: float = Field(default=0.9, description="Adam beta1.")
#     beta2: float = Field(default=0.99, description="Adam beta2.")
#     eps: float = Field(default=1e-3, description="Adam epsilon (tau).")


# class FeAmDPTrainingTaskConfig(TrainingTaskConfig):
#     fl_config: FLConfig
#     dp_config: DPConfig
#     server_optim: ServerAdamConfig = ServerAdamConfig()
