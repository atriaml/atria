from __future__ import annotations

from typing import TYPE_CHECKING

from atria_ml.configs._task import TaskConfigBase
from atria_ml.optimizers._configs import AdamOptimizerConfig, OptimizerConfigType
from atria_ml.training._configs import ModelCheckpointConfig
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from pydantic import ConfigDict

from atria_prv.dp.configs import DPConfig
from atria_prv.fl.configs import FLConfig

if TYPE_CHECKING:
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline


class FeAmDPTrainingTaskConfig(TaskConfigBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    model_pipeline: ModelPipelineConfig
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()
    do_train: bool = True
    do_test: bool = True
    do_validation: bool = True
    test_run: bool = False
    validate_every_n_rounds: float = 1.0
    fl_config: FLConfig
    dp_config: DPConfig
    global_optimizer: OptimizerConfigType = AdamOptimizerConfig()


class FeAmDPClientTrainingTaskConfig(TaskConfigBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    model_pipeline: ModelPipelineConfig
    test_run: bool = False
    dp_config: DPConfig
    client_id: int
    partition_cache_dir: str

    def build_client(self, model_pipeline: ModelPipeline | None = None):
        from atria_prv.feamdp._trainers._feam_dp_client_trainer import (
            FeAmDPClientTrainer,
        )

        return FeAmDPClientTrainer(config=self, model_pipeline=model_pipeline)
