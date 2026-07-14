from typing import Literal

from atria_ml.configs._task import TaskConfigBase
from atria_ml.configs._trainer import TrainerConfig
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from pydantic import BaseModel


class AttackConfig(BaseModel):
    """Configuration for the loss-based ART membership inference attack."""

    # ART attack model trained on the per-document loss feature.
    attack_model_type: Literal["rf", "gb", "nn", "lr"] = "nn"
    # AttackDataPipeline attack-train / attack-test split ratio (ART attack_train_size).
    attack_train_ratio: float = 0.5
    # Worst-case membership advantage: report the attack TPR at this target FPR.
    targeted_fpr: float = 0.01


class ShadowConfig(BaseModel):
    """Configuration for the shadow-model membership inference attack (Shokri et al.).

    When ``enabled`` is False the attack runs in the original "direct" mode (the attack
    model is trained on the target model's own member/non-member features). When enabled,
    ``num_shadow_models`` shadow models are trained on subsampled halves of the target
    train split and the attack model is trained purely on their known in/out behaviour;
    the target model's data is then used only for evaluation.
    """

    # Turn shadow-model mode on/off. Off => the original direct attack, unchanged.
    enabled: bool = False
    # Number of shadow models to train.
    num_shadow_models: int = 3
    # Fraction of the train original-ids used as the "in" (member) set per shadow;
    # the complementary half is the "out" (non-member) set.
    in_ratio: float = 0.5
    # Base RNG seed; shadow i uses ``seed + i`` for its in/out split.
    seed: int = 42
    # Reuse an existing shadow checkpoint instead of retraining when one is found.
    reuse_checkpoints: bool = True
    # Training hyperparameters for each shadow model (should mirror how the target was
    # trained). The shadow architecture comes from ``model_pipeline``.
    trainer: TrainerConfig = TrainerConfig()


class MembershipInferenceTaskConfig(TaskConfigBase):
    """Task config for running a membership inference attack against a trained model."""

    model_pipeline: ModelPipelineConfig
    target_checkpoint: str
    attack_config: AttackConfig = AttackConfig()
    shadow_config: ShadowConfig = ShadowConfig()
    with_amp: bool = False
    test_run: bool = False
