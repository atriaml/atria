"""ModelHubOps - snapshot save/load and hub upload/download for ModelPipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_hub.hub import AtriaHubConnectionError
from atria_logger import get_logger
from atria_registry._module_base import ModuleConfig
from atriax_client.models.task_type import TaskType

from atria_models.core.model_pipelines.constants import (
    _DEFAULT_ATRIA_MODELS_STORAGE_SUBDIR,
    _DEFAULT_MODEL_METADATA_PATH,
    _DEFAULT_MODEL_WEIGHTS_PATH,
    DEFAULT_ATRIA_MODELS_CACHE_DIR,
)

if TYPE_CHECKING:
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

logger = get_logger(__name__)


@dataclass
class SnapshotArtifact:
    weights: bytes
    metadata: bytes


class SafeTupleLoader(yaml.SafeLoader):
    pass


def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


SafeTupleLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", construct_python_tuple
)


class ModelHubOps:
    """Hub operations for ModelPipeline."""

    def __init__(self, pipeline: ModelPipeline) -> None:
        self._pipeline = pipeline

    def _default_snapshot_dir(self) -> Path:
        model_name = self._pipeline.config.model.model_name_or_path
        name = f"{model_name}-{self._pipeline.__pipeline_name__}"

        sanitized_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

        return Path(DEFAULT_ATRIA_MODELS_CACHE_DIR) / sanitized_name

    def create_snapshot(self) -> SnapshotArtifact:
        """
        Create an in-memory snapshot.

        Does NOT write anything to disk.
        """
        from safetensors.torch import save

        labels = self._pipeline._labels
        config = self._pipeline.config.to_dict()

        metadata = yaml.dump(
            {
                "config": config,
                "labels": (labels.model_dump() if labels is not None else None),
            }
        ).encode("utf-8")

        weights = save(self._pipeline._model.state_dict())

        return SnapshotArtifact(weights=weights, metadata=metadata)

    def save_snapshot(self, snapshot_dir: str | Path | None = None) -> Path:
        """
        Persist a snapshot to disk.
        """
        artifact = self.create_snapshot()

        target_dir = (
            Path(snapshot_dir)
            if snapshot_dir is not None
            else self._default_snapshot_dir()
        )

        target_dir.mkdir(parents=True, exist_ok=True)

        with open(target_dir / _DEFAULT_MODEL_WEIGHTS_PATH, "wb") as f:
            f.write(artifact.weights)

        with open(target_dir / _DEFAULT_MODEL_METADATA_PATH, "wb") as f:
            f.write(artifact.metadata)

        logger.info(f"Saved model snapshot to '{target_dir}'.")

        return target_dir

    def upload_to_hub(
        self,
        name: str | None = None,
        branch: str = "main",
        is_public: bool = False,
        overwrite_existing: bool = False,
    ) -> dict:
        from atria_hub.hub import AtriaHub

        try:
            snapshot = self.create_snapshot()

            hub_name = name or self._pipeline.config.model.model_name_or_path

            hub = AtriaHub().initialize()

            model_info = hub.models.get_or_create(
                username=str(hub.auth.username),
                name=hub_name,
                task_type=TaskType(self._pipeline.__pipeline_name__),
                is_public=is_public,
            )

            hub.models.upload_snapshot(
                model=model_info,
                branch=branch,
                config_name="default",
                files={
                    _DEFAULT_MODEL_WEIGHTS_PATH: snapshot.weights,
                    _DEFAULT_MODEL_METADATA_PATH: snapshot.metadata,
                },
                overwrite_existing=overwrite_existing,
            )

            logger.info(
                f"Model '{hub_name}' uploaded successfully to branch '{branch}'."
            )

            return {"username": hub.auth.username, "name": hub_name, "branch": branch}

        except AtriaHubConnectionError:
            logger.error(
                "Failed to connect to AtriaHub. "
                "Please check your connection and try again."
            )
            raise

        except Exception as e:
            logger.error(f"Failed to upload model to hub: {e}")
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
            "Expected 'model_name' or "
            "'username/model_name'."
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

        return cls.load_from_snapshot(target_path)

    @staticmethod
    def load_from_snapshot(snapshot_dir: Path) -> ModelPipeline:
        from atria_types import DatasetLabels
        from safetensors.torch import load_file

        with open(snapshot_dir / _DEFAULT_MODEL_METADATA_PATH) as f:
            metadata = yaml.load(f, Loader=SafeTupleLoader)

        config_dict = metadata.get("config")

        if config_dict is None:
            raise ValueError(
                f"Model metadata at '{snapshot_dir}' is missing 'config' field."
            )

        labels = metadata.get("labels")
        labels = DatasetLabels.model_validate(labels) if labels else DatasetLabels()

        config = ModuleConfig.from_dict(config_dict)

        pipeline = config.build(labels=labels)

        weights = load_file(snapshot_dir / _DEFAULT_MODEL_WEIGHTS_PATH)

        pipeline._model.load_state_dict(weights, strict=True)

        logger.info(f"Loaded model pipeline from snapshot at '{snapshot_dir}'.")

        return pipeline
