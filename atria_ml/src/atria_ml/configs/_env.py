from __future__ import annotations

from pathlib import Path

import codename
from atria_logger import get_logger
from atria_registry._module_base import BaseModel
from atria_types._utilities._repr import RepresentationMixin
from pydantic import ConfigDict, Field

logger = get_logger(__name__)


class RuntimeEnvConfig(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    project_name: str = "atria_ml"
    exp_name: str = Field(default_factory=lambda: codename.codename())
    dataset_name: str | None = None
    model_name: str | None = None
    output_dir: str = "???"
    seed: int = 42
    deterministic: bool = False
    backend: str | None = "nccl"
    n_devices: int = 1

    @property
    def run_dataset_dir(self) -> Path:
        base_path = Path(self.output_dir) / self.exp_name
        if self.dataset_name is not None:
            base_path = base_path / self.dataset_name
        return base_path

    @property
    def run_dir(self) -> Path:
        base_path = Path(self.output_dir) / self.exp_name
        if self.dataset_name is not None:
            base_path = base_path / self.dataset_name
        if self.model_name is not None:
            base_path = base_path / self.model_name
        return base_path
