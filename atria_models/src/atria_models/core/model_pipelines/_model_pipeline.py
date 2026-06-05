from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, Self

from atria_logger import get_logger
from atria_registry._module_base import ConfigurableModule, ModuleConfig
from atria_transforms.core._data_types._base import TensorDataModel
from atria_types._datasets import DatasetLabels

from atria_models.core.model_pipelines._common import T_ModelPipelineConfig
from atria_models.core.model_pipelines._ops import ModelPipelineOps
from atria_models.core.model_pipelines.constants import (
    _DEFAULT_MODEL_METADATA_PATH,
    _DEFAULT_MODEL_WEIGHTS_PATH,
    DEFAULT_ATRIA_MODELS_CACHE_DIR,
)
from atria_models.core.types.model_outputs import ModelOutput

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine
    from ignite.metrics import Metric


logger = get_logger(__name__)


@dataclass
class SnapshotArtifact:
    weights: bytes
    metadata: bytes


class ModelPipeline(
    ConfigurableModule[T_ModelPipelineConfig], Generic[T_ModelPipelineConfig]
):
    __abstract__ = True
    __config__: type[T_ModelPipelineConfig]
    __repr_fields__: set[str] = {"config", "labels", "model"}
    __pipeline_name__: ClassVar[str]

    def __init__(self, config: T_ModelPipelineConfig, labels: DatasetLabels) -> None:
        from atria_models.core.model_builders._base import ModelBuilder

        super().__init__(config=config)

        self._labels = labels
        self._model_builder = ModelBuilder.from_type(
            builder_type=self.config.model.builder_type,
            bn_to_gn=self.config.model.bn_to_gn,
            frozen_layers=self.config.model.frozen_layers,
            pretrained_checkpoint=self.config.model.pretrained_checkpoint,
            model_type=self.config.model.model_type,
        )

        assert self.config.model.model_name_or_path is not None, (
            "ModelPipelineConfig.model.model_name_or_path must be specified to build the model."
            f" Current config: {self.config}"
        )
        self._model = self._model_builder.build(
            model_name_or_path=self.config.model.model_name_or_path,
            **(self.config.model.model_kwargs or {}),
            **self._model_build_kwargs(),
        )
        self._model_args_list = []
        for param in inspect.signature(self._model.forward).parameters.values():
            if param.name != "self":
                self._model_args_list.append(param.name)

    @property
    def ops(self) -> ModelPipelineOps:
        return ModelPipelineOps(self)

    @property
    def ema_modules(self) -> torch.nn.Module:
        return self._model

    @property
    def trainable_parameters(self) -> dict[str, Iterator[torch.nn.Parameter]]:
        return self.ops.get_trainable_parameters()

    def _model_build_kwargs(self) -> dict[str, object]:
        return {}

    def build_metrics(
        self,
        stage: Literal["train", "validation", "test"],
        device: torch.device | str = "cpu",
    ) -> dict[str, Metric]:
        return {}

    def training_step(
        self, batch: TensorDataModel, training_engine: Engine | None = None, **kwargs
    ) -> ModelOutput:
        raise NotImplementedError(
            "Training step is not implemented for this model pipeline."
        )

    def evaluation_step(
        self,
        batch: TensorDataModel,
        evaluation_engine: Engine | None = None,
        training_engine: Engine | None = None,
        stage: Literal["validation", "test"] = "test",
        **kwargs,
    ) -> ModelOutput:
        raise NotImplementedError(
            "Evaluation step is not implemented for this model pipeline."
        )

    def predict_step(
        self, batch: TensorDataModel, prediction_engine: Engine | None = None, **kwargs
    ) -> ModelOutput:
        raise NotImplementedError(
            "Prediction step is not implemented for this model pipeline."
        )

    def visualization_step(
        self,
        batch: TensorDataModel,
        visualization_engine: Engine | None = None,
        training_engine: Engine | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            "Visualization step is not implemented for this model pipeline."
        )

    def state_dict(self) -> dict:
        return {"config": self.config.to_dict(), "model": self._model.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        if "config" in state_dict:
            self.config.model_validate(state_dict["config"])
        if "model" in state_dict:
            self._model.load_state_dict(state_dict["model"], strict=True)

    def snapshot(self) -> SnapshotArtifact:
        import yaml
        from safetensors.torch import save

        labels = self._labels
        config = self.config.to_dict()

        metadata = yaml.dump(
            {
                "pipeline_name": self.__pipeline_name__,
                "config": config,
                "labels": (labels.model_dump() if labels is not None else None),
            }
        ).encode("utf-8")

        weights = save(self._model.state_dict())

        return SnapshotArtifact(weights=weights, metadata=metadata)

    @classmethod
    def from_snapshot(cls, snapshot: SnapshotArtifact) -> Self:
        import yaml
        from atria_types import DatasetLabels
        from safetensors.torch import load

        metadata = yaml.safe_load(snapshot.metadata.decode("utf-8"))
        config_dict = metadata.get("config")
        labels = metadata.get("labels")

        if config_dict is None:
            raise ValueError("Model metadata is missing 'config' field.")
        if labels is None:
            raise ValueError("Model metadata is missing 'labels' field.")

        labels = DatasetLabels.model_validate(labels)
        config = ModuleConfig.from_dict(config_dict)
        pipeline = config.build(labels=labels)
        pipeline._model.load_state_dict(load(snapshot.weights), strict=True)
        return pipeline

    def load_checkpoint(self, checkpoint_path: str) -> None:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert "model_pipeline" in checkpoint, (
            f"Checkpoint at {checkpoint_path} does not contain 'model_pipeline' key."
        )
        self.load_state_dict(checkpoint["model_pipeline"])

    def _default_snapshot_dir(self) -> Path:
        model_name = self.config.model.model_name_or_path
        name = f"{model_name}-{self.__pipeline_name__}"

        sanitized_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

        return Path(DEFAULT_ATRIA_MODELS_CACHE_DIR) / sanitized_name

    def save_to_disk(self, snapshot_dir: str | Path | None = None) -> Path:
        artifact = self.snapshot()
        target_dir = (
            Path(snapshot_dir)
            if snapshot_dir is not None
            else self._default_snapshot_dir()
        )

        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(target_dir / _DEFAULT_MODEL_WEIGHTS_PATH, "wb") as f:
                f.write(artifact.weights)

            with open(target_dir / _DEFAULT_MODEL_METADATA_PATH, "wb") as f:
                f.write(artifact.metadata)
        except Exception:
            for path in [
                target_dir / _DEFAULT_MODEL_WEIGHTS_PATH,
                target_dir / _DEFAULT_MODEL_METADATA_PATH,
            ]:
                if path.exists():
                    path.unlink()

            if target_dir.exists() and not any(target_dir.iterdir()):
                target_dir.rmdir()

            raise

        logger.info(f"Saved model snapshot to '{target_dir}'.")
        return target_dir

    @classmethod
    def load_from_disk(cls, snapshot_dir: str | Path) -> Self:
        snapshot_dir = Path(snapshot_dir)
        if not snapshot_dir.exists():
            raise FileNotFoundError(
                f"Snapshot directory '{snapshot_dir}' does not exist."
            )
        if not snapshot_dir.is_dir():
            raise ValueError(f"Snapshot path '{snapshot_dir}' is not a directory.")

        weights_path = snapshot_dir / _DEFAULT_MODEL_WEIGHTS_PATH
        metadata_path = snapshot_dir / _DEFAULT_MODEL_METADATA_PATH

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights file '{weights_path}' does not exist."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata file '{metadata_path}' does not exist."
            )

        with open(weights_path, "rb") as f:
            weights = f.read()
        with open(metadata_path, "rb") as f:
            metadata = f.read()

        snapshot = SnapshotArtifact(weights=weights, metadata=metadata)
        return cls.from_snapshot(snapshot)

    def upload_to_hub(
        self,
        name: str,
        branch: str = "main",
        is_public: bool = False,
        overwrite_existing: bool = False,
    ) -> dict:
        from atria_models.core.model_pipelines._hub_ops import ModelHubOps

        return ModelHubOps(self).upload_to_hub(
            name=name,
            branch=branch,
            is_public=is_public,
            overwrite_existing=overwrite_existing,
        )

    @classmethod
    def load_from_hub(
        cls, name: str, branch: str = "main", download_dir=None
    ) -> ModelPipeline:
        from atria_models.core.model_pipelines._hub_ops import ModelHubOps

        return ModelHubOps.load_from_hub(
            name=name, branch=branch, download_dir=download_dir
        )
