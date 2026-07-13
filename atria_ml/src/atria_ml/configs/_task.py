from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

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

METRICS_SCHEMA_VERSION = 1

logger = get_logger(__name__)


class TaskConfigBase(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    env: RuntimeEnvConfig = Field(default_factory=RuntimeEnvConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        results = super().model_dump()
        preprocess_train_transform = results["data"]["preprocess_train_transform"]
        if preprocess_train_transform is not None and callable(
            preprocess_train_transform
        ):
            results["data"]["preprocess_train_transform"] = (
                preprocess_train_transform.__name__
            )
        preprocess_eval_transform = results["data"]["preprocess_eval_transform"]
        if preprocess_eval_transform is not None and callable(
            preprocess_eval_transform
        ):
            results["data"]["preprocess_eval_transform"] = (
                preprocess_eval_transform.__name__
            )
        return results

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

    def dump_metrics_file(
        self, data: dict, started_at: datetime | None = None, extra: dict | None = None
    ) -> None:
        completed_at = datetime.now(timezone.utc)
        output_file_path = self.get_metrics_file_path()
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "run": {
                "id": str(uuid.uuid4()),
                "started_at": started_at.isoformat() if started_at else None,
                "completed_at": completed_at.isoformat(),
                "duration_seconds": (
                    (completed_at - started_at).total_seconds() if started_at else None
                ),
            },
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "hostname": socket.gethostname(),
                "cwd": os.getcwd(),
                "argv": sys.argv,
            },
            "config": self.model_dump(),
            "data": data,
            **(extra or {}),
        }

        tmp = output_file_path.with_suffix(output_file_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4, default=str)
        tmp.replace(output_file_path)

        logger.info(f"Metrics dumped to {output_file_path}")


class TrainingTaskConfig(TaskConfigBase):
    model_pipeline: ModelPipelineConfig
    trainer: TrainerConfig = TrainerConfig()
    do_train: bool = True
    do_test: bool = True
    do_validation: bool = True
    do_visualization: bool = False
    reevaluate_metrics: bool = True
    test_run: bool = False
    use_fixed_batch_iterator: bool = False
    save_test_outputs_to_disk: bool = False
    use_ema_for_evaluation: bool = False
    with_amp: bool = True


class EvaluationTaskConfig(TaskConfigBase):
    model_pipeline: ModelPipelineConfig
    eval_checkpoint: str
    save_snapshot: bool = True
    test_run: bool = False
    use_fixed_batch_iterator: bool = False
    save_test_outputs_to_disk: bool = False
    use_ema_for_evaluation: bool = False
    with_amp: bool = True

    @classmethod
    def from_training_config(
        cls, training_config: TrainingTaskConfig, eval_checkpoint: str
    ) -> Self:
        return cls(
            env=training_config.env,
            data=training_config.data,
            logging=training_config.logging,
            test_run=training_config.test_run,
            use_fixed_batch_iterator=training_config.use_fixed_batch_iterator,
            save_test_outputs_to_disk=training_config.save_test_outputs_to_disk,
            use_ema_for_evaluation=training_config.use_ema_for_evaluation,
            with_amp=training_config.with_amp,
            model_pipeline=training_config.model_pipeline,
            eval_checkpoint=eval_checkpoint,
        )
