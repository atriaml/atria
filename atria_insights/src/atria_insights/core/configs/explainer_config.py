from __future__ import annotations

import hashlib
import json
from pathlib import Path

from atria_datasets.core.dataset._datasets import Dataset
from atria_logger import get_logger
from atria_ml.configs._base import DataConfig, RuntimeEnvConfig, pydantic_to_hydra
from atria_ml.training._configs import LoggingConfig
from atria_registry._module_base import BaseModel
from atria_types._utilities._repr import RepresentationMixin
from pydantic import ConfigDict, Field

from atria_insights.core.model_pipelines._common import ExplainableModelPipelineConfig

logger = get_logger(__name__)


class ExplainerRunConfig(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    env: RuntimeEnvConfig = Field(default_factory=RuntimeEnvConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    x_model_pipeline: ExplainableModelPipelineConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    test_run: bool = False
    use_fixed_batch_iterator: bool = False
    save_test_outputs_to_disk: bool = False
    use_ema_for_evaluation: bool = False
    with_amp: bool = False

    def build_dataset(self) -> Dataset:
        return self.data.build_dataset()

    def state_dict(self) -> dict:
        return self.model_dump()

    def load_state_dict(self, state_dict: dict) -> None:
        self.model_validate(state_dict)

    def get_metrics_file_path(self) -> Path:
        params = self.model_dump()
        config_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:8]
        return Path(self.env.run_dir) / f"outputs-{config_hash}.json"

    def metrics_file_exists(self) -> bool:
        output_file_path = self.get_metrics_file_path()
        return output_file_path.exists()

    def dump_metrics_file(self, data: dict) -> None:
        output_file_path = self.get_metrics_file_path()
        with open(output_file_path, "w") as f:
            json.dump({"config": self.model_dump(), "data": data}, f, indent=4)
        logger.info(f"Metrics dumped to {output_file_path}")

    def save_to_json(self, file_path: str | Path | None = None) -> None:
        if file_path is None:
            file_path = Path(self.env.run_dir) / "config.json"
        else:
            file_path = Path(file_path)

        # make parent
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use pydantic_to_hydra to convert to Hydra-compatible format
        hydra_data = pydantic_to_hydra(self)

        with open(file_path, "w") as f:
            json.dump(hydra_data, f, indent=4)

        logger.info(f"RunConfig saved to {file_path}")

    @classmethod
    def from_json(cls, file_path: str | Path) -> ExplainerRunConfig:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        file_path = Path(file_path)
        with open(file_path) as f:
            data = json.load(f)

        # Convert to OmegaConf
        omega_conf = OmegaConf.create(data)

        # Use Hydra instantiate to create the object
        return instantiate(omega_conf)
