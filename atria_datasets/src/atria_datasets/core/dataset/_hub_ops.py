"""Dataset Hub Operations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from atria_hub.api.datasets import FilesExistError
from atria_hub.hub import (
    AtriaHub,  # type: ignore[import-not-found]
    AtriaHubConnectionError,
)
from atria_logger import get_logger

from atria_datasets.core.constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR,
)

if TYPE_CHECKING:
    from atria_datasets.core.dataset._cached_dataset import CachedDataset

logger = get_logger(__name__)


def _data_model_to_instance_type(data_model):
    try:
        from atriax_client.models.data_instance_type import (  # type: ignore[import-not-found]
            DataInstanceType,
        )
    except ImportError:
        raise ImportError(
            "The 'atriax_client' package is required. Install with: uv add atriax_client"
        )
    from atria_types import DocumentInstance, ImageInstance

    if data_model is ImageInstance:
        return DataInstanceType.IMAGE_INSTANCE
    elif data_model is DocumentInstance:
        return DataInstanceType.DOCUMENT_INSTANCE
    else:
        raise ValueError(f"Unsupported data model for hub upload: {data_model}")


class DatasetHubOps:
    """Hub operations available only on CachedDataset."""

    def __init__(self, dataset: CachedDataset):
        self._dataset = dataset

    def prepare_dataset_files_from_dir(self) -> list[tuple[str, str]]:
        """Collect all (source, target) file pairs from the cached dataset directory."""
        path = self._dataset._path
        if not path.exists():
            raise RuntimeError(
                f"Cached dataset directory not found: {path}. "
                "Build the dataset first using cache()."
            )
        return [
            (str(f), str(f.relative_to(path.parent)))
            for f in path.rglob("*")
            if f.is_file()
        ]

    def upload_to_hub(
        self,
        name: str | None = None,
        branch: str = "main",
        is_public: bool = False,
        overwrite_existing: bool = False,
    ) -> dict[str, str]:
        """Upload the cached dataset to Atria Hub.

        Args:
            name: Hub dataset name (defaults to dataset config name)
            branch: Hub branch to upload to
            is_public: Whether to make the dataset public
            overwrite_existing: Overwrite existing files on hub
        """

        hub_name = (
            name or self._dataset.dataset_name or self._dataset.dataset_class_name
        )

        try:
            hub = AtriaHub().initialize()
            dataset_info = hub.datasets.get_or_create(
                username=str(hub.auth.username),
                name=hub_name,
                default_branch=branch,
                description=self._dataset.metadata.description
                if self._dataset.metadata
                else None,
                data_instance_type=_data_model_to_instance_type(
                    self._dataset.data_model
                ),
                is_public=is_public,
            )
            try:
                hub.datasets.upload_files(
                    dataset=dataset_info,
                    branch=branch,
                    config_dir=self._dataset._path.name,
                    dataset_files=self.prepare_dataset_files_from_dir(),
                    overwrite_existing=overwrite_existing,
                )
                logger.info(
                    f"Dataset '{hub_name}' uploaded successfully to branch '{branch}'."
                )
            except FilesExistError:
                logger.warning(
                    f"Files already exist in dataset '{hub_name}' on branch '{branch}'. "
                    "Set overwrite_existing=True to overwrite existing files."
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
    def _parse_dataset_name(name: str) -> tuple[str | None, str]:
        parts = name.split("/")
        if len(parts) == 1:
            return None, parts[0]
        if len(parts) == 2:
            return parts[0], parts[1]
        raise ValueError(
            f"Invalid dataset name format: {name}. "
            "Expected 'dataset_name' or 'username/dataset_name'."
        )

    @staticmethod
    def _resolve_config_name(
        hub, dataset_repo_id: str, branch: str, config_name: str | None
    ) -> str:
        available_configs = hub.datasets.get_available_configs(
            dataset_repo_id, branch=branch
        )
        if config_name is None:
            if len(available_configs) == 1:
                return available_configs[0]
            if len(available_configs) == 0:
                raise ValueError(
                    f"No configurations available for repository '{dataset_repo_id}' on branch '{branch}'."
                )
            raise RuntimeError(
                "Multiple configurations are available. "
                f"Please pass config_dir explicitly. Available configurations: {available_configs}"
            )

        if config_name in available_configs:
            return config_name

        matching_configs = [
            cfg for cfg in available_configs if cfg.startswith(config_name)
        ]
        if len(matching_configs) == 1:
            logger.info(
                f"Resolved config_dir '{config_name}' to '{matching_configs[0]}'."
            )
            return matching_configs[0]

        if len(matching_configs) > 1:
            raise RuntimeError(
                f"Multiple configurations match '{config_name}': {matching_configs}. "
                "Please specify config_dir explicitly."
            )

        raise ValueError(
            f"Configuration '{config_name}' not found on branch '{branch}'. "
            f"Available configurations: {available_configs}"
        )

    @classmethod
    def load_from_hub(
        cls,
        name: str,
        username: str | None = None,
        branch: str = "main",
        config_name: str | None = None,
        storage_dir: str | Path | None = None,
        overwrite_existing: bool = False,
    ) -> CachedDataset:
        """Download a frozen cached dataset snapshot from Atria Hub.

        Args:
            name: Dataset name in format 'dataset_name' or 'username/dataset_name'.
            username: Dataset owner username when `name` does not include a username.
            branch: Hub branch to download from.
            config_name: Frozen cached configuration name on the hub.
                If omitted and exactly one config exists, that config is used.
            storage_dir: Local storage directory where the config directory should be placed.
                Defaults to '~/.cache/atria/datasets/<dataset_name>/storage'.
            overwrite_existing: Overwrite an already downloaded local snapshot.

        Returns:
            CachedDataset: Loaded file-backed cached dataset.
        """
        try:
            from atria_hub.hub import AtriaHub  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to download datasets from the hub. "
                "Install with: uv add https://github.com/saifullah3396/atria_hub"
            )
        from atria_datasets.core.dataset._cached_dataset import CachedDataset

        dataset_owner_from_name, dataset_name = cls._parse_dataset_name(name)
        if dataset_owner_from_name is not None and username is not None:
            if dataset_owner_from_name != username:
                raise ValueError(
                    "Conflicting usernames provided. "
                    f"Got '{dataset_owner_from_name}' in name and '{username}' in username."
                )

        hub = AtriaHub().initialize()
        owner = dataset_owner_from_name or username or str(hub.auth.username)
        dataset_info = hub.datasets.get_by_name(username=owner, name=dataset_name)
        if dataset_info is None or not hasattr(dataset_info, "repo_id"):
            raise RuntimeError(
                f"Failed to resolve dataset '{owner}/{dataset_name}' from the hub."
            )
        dataset_repo_id = str(cast(Any, dataset_info).repo_id)

        resolved_config_dir = cls._resolve_config_name(
            hub=hub,
            dataset_repo_id=dataset_repo_id,
            branch=branch,
            config_name=config_name,
        )

        base_storage_dir = (
            Path(storage_dir)
            if storage_dir is not None
            else (
                Path(_DEFAULT_ATRIA_DATASETS_CACHE_DIR)
                / dataset_name
                / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
            )
        )
        target_path = base_storage_dir / resolved_config_dir

        if target_path.exists():
            if not overwrite_existing:
                logger.info(
                    f"Cached dataset already exists at '{target_path}'. "
                    "Skipping download."
                )
                return CachedDataset(path=target_path).load()
            logger.warning(f"Overwriting existing cached dataset at '{target_path}'.")
            shutil.rmtree(target_path)

        base_storage_dir.mkdir(parents=True, exist_ok=True)
        hub.datasets.download_files(
            dataset_repo_id=dataset_repo_id,
            branch=branch,
            config_dir=resolved_config_dir,
            destination_path=str(base_storage_dir),
        )

        if not target_path.exists():
            raise RuntimeError(
                "Dataset download finished but expected cached path was not created: "
                f"{target_path}"
            )

        logger.info(
            f"Dataset '{owner}/{dataset_name}' downloaded successfully to '{target_path}'."
        )
        return CachedDataset(path=target_path).load()
