from typing import Literal

from atria_ml.configs._task import TaskConfigBase
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from pydantic import BaseModel


class AttackConfig(BaseModel):
    """Configuration for the loss-based ART membership inference attack."""

    # ART attack model trained on the per-document loss feature.
    attack_model_type: Literal["rf", "gb", "nn", "lr", "mlp"] = "nn"
    # AttackDataPipeline attack-train / attack-test split ratio (ART attack_train_size).
    attack_train_ratio: float = 0.5
    # Worst-case membership advantage: report the attack TPR at this target FPR.
    targeted_fpr: float = 0.01


class MembershipInferenceTaskConfig(TaskConfigBase):
    """Task config for running a membership inference attack against a trained model."""

    model_pipeline: ModelPipelineConfig
    target_checkpoint: str
    attack_config: AttackConfig = AttackConfig()
    with_amp: bool = False
    test_run: bool = False
