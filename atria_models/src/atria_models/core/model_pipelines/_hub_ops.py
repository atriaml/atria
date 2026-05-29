"""ModelHubOps - snapshot save/load and hub upload/download for ModelPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_hub.src.atria_hub.hub import AtriaHubConnectionError
from atria_logger import get_logger
from atria_registry._module_base import ModuleConfig

from atria_models.core.model_pipelines.constants import (
    _DEFAULT_ATRIA_MODELS_STORAGE_SUBDIR,
    _DEFAULT_MODEL_METADATA_PATH,
    _DEFAULT_MODEL_WEIGHTS_PATH,
    DEFAULT_ATRIA_MODELS_CACHE_DIR,
)

if TYPE_CHECKING:
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

logger = get_logger(__name__)


class ModelHubOps:
    """Hub operations for ModelPipeline — mirrors DatasetHubOps pattern."""

    def __init__(self, pipeline: ModelPipeline) -> None:
        self._pipeline = pipeline

    @staticmethod
    def _prepare_snapshot_files(snapshot_dir: Path) -> list[tuple[str, str]]:
        return [
            (str(f), str(f.relative_to(snapshot_dir.parent)))
            for f in snapshot_dir.rglob("*")
            if f.is_file()
        ]

    def save_snapshot(
        self, name: str | None = None, config_name: str = "default"
    ) -> Path:
        from safetensors.torch import save_file

        if name is None:
            model_name = self._pipeline.config.model.model_name_or_path
            name = f"{model_name}_{self._pipeline.__pipeline_name__}"

        snapshot_dir = (
            Path(DEFAULT_ATRIA_MODELS_CACHE_DIR)
            / name
            / _DEFAULT_ATRIA_MODELS_STORAGE_SUBDIR
            / config_name
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        save_file(
            self._pipeline._model.state_dict(),
            snapshot_dir / _DEFAULT_MODEL_WEIGHTS_PATH,
        )

        labels = self._pipeline._labels
        config = self._pipeline.config.to_dict()

        with open(snapshot_dir / _DEFAULT_MODEL_METADATA_PATH, "w") as f:
            yaml.dump(
                {
                    "config": config,
                    "labels": labels.model_dump() if labels is not None else None,
                },
                f,
            )

        logger.info(f"Saved model snapshot to '{snapshot_dir}'.")
        return snapshot_dir

    def upload_to_hub(
        self,
        name: str | None = None,
        branch: str = "main",
        is_public: bool = False,
        overwrite_existing: bool = False,
    ) -> dict:
        from atria_hub.hub import AtriaHub

        try:
            snapshot_dir = self.save_snapshot(name)
            hub_name = snapshot_dir.parent.parent.name

            hub = AtriaHub().initialize()
            model_info = hub.models.get_or_create(
                username=str(hub.auth.username),
                name=hub_name,
                task_type=self._pipeline.__pipeline_name__,
                is_public=is_public,
            )
            hub.models.upload_files(
                model=model_info,
                branch=branch,
                config_name=snapshot_dir.name,
                model_files=self._prepare_snapshot_files(snapshot_dir),
                overwrite_existing=overwrite_existing,
            )
            logger.info(
                f"Model '{hub_name}' uploaded successfully to branch '{branch}'."
            )
            return {"username": hub.auth.username, "name": hub_name, "branch": branch}

        except AtriaHubConnectionError:
            logger.error(
                "Failed to connect to AtriaHub. Please check your connection and try again."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to upload dataset to hub: {e}")
            raise

    @staticmethod
    def _parse_model_name(name: str) -> tuple[str | None, str]:
        parts = name.split("/")
        if len(parts) == 1:
            return None, parts[0]
        if len(parts) == 2:
            return parts[0], parts[1]
        raise ValueError(
            f"Invalid model name format: '{name}'. "
            "Expected 'model_name' or 'username/model_name'."
        )

    @classmethod
    def load_from_hub(
        cls,
        name: str,
        branch: str = "main",
        config_name: str = "default",
        download_dir: str | Path | None = None,
    ) -> ModelPipeline:
        from atria_hub.hub import AtriaHub

        username, model_name = cls._parse_model_name(name)
        hub = AtriaHub().initialize()
        owner = username or str(hub.auth.username)
        model_info = hub.models.get_by_name(owner, model_name)

        dest = (
            Path(download_dir)
            if download_dir is not None
            else Path(DEFAULT_ATRIA_MODELS_CACHE_DIR)
            / model_name
            / _DEFAULT_ATRIA_MODELS_STORAGE_SUBDIR
        )
        target_path = dest / config_name

        if target_path.exists():
            logger.info(
                f"Cached model snapshot exists at '{target_path}'. Skipping download."
            )
        else:
            dest.mkdir(parents=True, exist_ok=True)
            hub.models.download_files(
                model_repo_id=str(model_info.repo_id),
                branch=branch,
                config_name=config_name,
                destination_path=str(dest),
            )

        return cls._load_from_snapshot(target_path)

    @staticmethod
    def _load_from_snapshot(snapshot_dir: Path) -> ModelPipeline:
        from atria_types import DatasetLabels
        from safetensors.torch import load_file

        with open(snapshot_dir / _DEFAULT_MODEL_METADATA_PATH) as f:
            metadata = yaml.safe_load(f)

        config_dict = metadata.get("config")
        if config_dict is None:
            raise ValueError(
                f"Model metadata at '{snapshot_dir}' is missing 'config' field."
            )

        labels = metadata.get("labels")
        labels = DatasetLabels.model_validate(labels) if labels else DatasetLabels()

        config = ModuleConfig.from_dict(config_dict)  # Validate config dict structure
        pipeline = config.build(labels=labels)

        weights = load_file(snapshot_dir / _DEFAULT_MODEL_WEIGHTS_PATH)
        pipeline._model.load_state_dict(weights, strict=True)

        logger.info(f"Loaded model pipeline from snapshot at '{snapshot_dir}'.")
        return pipeline
