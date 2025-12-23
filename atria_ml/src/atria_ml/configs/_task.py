from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Self

from atria_datasets.core.dataset._datasets import Dataset
from atria_logger import get_logger
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from atria_registry._module_base import BaseModel
from atria_registry._utilities import to_instantiable_dict
from atria_types._utilities._repr import RepresentationMixin
from pydantic import ConfigDict, Field

from atria_ml.configs._data import DataConfig
from atria_ml.configs._env import RuntimeEnvConfig
from atria_ml.configs._trainer import TrainerConfig
from atria_ml.training._configs import LoggingConfig

logger = get_logger(__name__)


class TaskConfigBase(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    env: RuntimeEnvConfig = Field(default_factory=RuntimeEnvConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    test_run: bool = False
    use_fixed_batch_iterator: bool = False
    save_test_outputs_to_disk: bool = False
    use_ema_for_evaluation: bool = False
    with_amp: bool = True

    def build_dataset(self) -> Dataset:
        return self.data.build_dataset()

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        omega_conf = OmegaConf.create(obj)
        obj = instantiate(omega_conf)
        return cls.model_validate(obj)

    def to_dict(self) -> dict:
        return to_instantiable_dict(self)

    def state_dict(self) -> dict:
        return self.to_dict()

    @classmethod
    def from_json(cls, file_path: str | Path) -> Self:
        file_path = Path(file_path)
        with open(file_path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_json(self, file_path: str | Path | None = None) -> None:
        if file_path is None:
            file_path = Path(self.env.run_dir) / "config.json"
        else:
            file_path = Path(file_path)

        # make parent
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

        logger.info(f"RunConfig saved to {file_path}")

    def get_metrics_file_path(self) -> Path:
        params = self.model_dump()
        config_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:8]
        return Path(self.env.run_dir) / "test" / f"{config_hash}.json"

    def metrics_file_exists(self) -> bool:
        output_file_path = self.get_metrics_file_path()
        return output_file_path.exists()

    def dump_metrics_file(self, data: dict) -> None:
        output_file_path = self.get_metrics_file_path()
        if not output_file_path.parent.exists():
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w") as f:
            json.dump({"config": self.model_dump(), "data": data}, f, indent=4)
        logger.info(f"Metrics dumped to {output_file_path}")


class TrainingTaskConfig(TaskConfigBase):
    model_pipeline: ModelPipelineConfig
    trainer: TrainerConfig = TrainerConfig()
    do_train: bool = True
    do_test: bool = True
    do_validation: bool = True
    do_visualization: bool = False
    reevaluate_metrics: bool = True
