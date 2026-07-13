from typing import Literal

from atria_ml.configs._task import TaskConfigBase
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from pydantic import BaseModel

AttackType = Literal["black_box", "rule_based"]


class AttackConfig(BaseModel):
    """Configuration for an ART membership inference attack (follows the ART tutorial)."""

    attack_type: AttackType = "black_box"
    # If the adversary has no access to the training set, the attack can't be fit on
    # real members -> falls back to the rule-based attack (needs no fitting).
    adversary_has_trainset_access: bool = True
    attack_model_type: Literal["rf", "gb", "nn"] = "rf"
    # ART attack_train_size ratio: fraction of members / non-members used to FIT the
    # attack (attack-train); the rest is held out to score it (attack-test).
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
