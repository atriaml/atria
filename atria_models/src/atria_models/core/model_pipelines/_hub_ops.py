"""ModelHubOps - snapshot save/load and hub upload/download for ModelPipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_hub.hub import AtriaHubConnectionError
from atria_logger import get_logger

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

    def upload_to_hub(
        self,
        name: str | None = None,
        branch: str = "main",
        is_public: bool = False,
        overwrite_existing: bool = False,
    ) -> dict:
        from atria_hub.hub import AtriaHub

        try:
            snapshot = self._pipeline.snapshot()
            hub_name = name or self._pipeline.config.model.model_name_or_path

            hub = AtriaHub().initialize()

            model_info = hub.models.get_or_create(
                username=str(hub.auth.username), name=hub_name, is_public=is_public
            )

            hub.models.upload_snapshot(
                model=model_info,
                branch=branch,
                files={
                    _DEFAULT_MODEL_WEIGHTS_PATH: snapshot.weights,
                    _DEFAULT_MODEL_METADATA_PATH: snapshot.metadata,
                },
                overwrite_existing=overwrite_existing,
            )
            hub.models.finalize(model=model_info, branch=branch)

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
        cls, name: str, branch: str = "main", download_dir: str | Path | None = None
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

        target_path = dest

        if target_path.exists():
            logger.info(
                f"Cached model snapshot exists at '{target_path}'. Skipping download."
            )
        else:
            dest.mkdir(parents=True, exist_ok=True)

            hub.models.download_files(
                model_repo_id=str(model_info.repo_id),
                branch=branch,
                destination_path=str(dest),
            )

        return cls.load_from_snapshot(target_path)
