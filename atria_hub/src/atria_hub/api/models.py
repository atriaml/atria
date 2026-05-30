from __future__ import annotations

from typing import TYPE_CHECKING

from atria_hub.api.base import BaseApi

if TYPE_CHECKING:
    import uuid

    from atriax_client.models.body_model_create import BodyModelCreate
    from atriax_client.models.model import Model


class ModelNotFoundError(Exception):
    """Custom exception for model not found errors."""

    pass


class ModelConfigNotFoundError(Exception):
    """Custom exception for model configuration not found errors."""

    pass


class InvalidModelConfigError(Exception):
    """Custom exception for invalid model configuration errors."""

    pass


class ModelsApi(BaseApi):
    def get(self, id: uuid.UUID) -> Model:
        """Retrieve a model from the hub by its name."""

        from atriax_client.api.model import model_item

        with self._client.protected_api_client as client:
            response = model_item.sync_detailed(id, client=client)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get model: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def get_by_name(self, username: str, name: str):
        """Retrieve a model from the hub by its name."""

        from atriax_client.api.model import model_find_one

        with self._client.protected_api_client as client:
            response = model_find_one.sync_detailed(
                client=client, username=username, name=name
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get model: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def create(self, body: BodyModelCreate):
        """Create a new model in the hub."""

        from atriax_client.api.model import model_create

        with self._client.protected_api_client as client:
            response = model_create.sync_detailed(client=client, body=body)
            if response.status_code not in [200, 201]:
                raise RuntimeError(
                    f"Failed to create model: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def get_or_create(
        self,
        username: str,
        name: str,
        default_branch: str = "main",
        description: str | None = None,
        is_public: bool = False,
    ) -> Model:
        """Get or create a model in the hub."""

        from atriax_client.models.body_model_create import BodyModelCreate

        try:
            return self.get_by_name(username=username, name=name)
        except Exception:
            return self.create(
                body=BodyModelCreate(
                    name=name,
                    default_branch=default_branch,
                    description=description,
                    is_public=is_public,
                )
            )

    def upload_snapshot(
        self,
        model: Model,
        branch: str,
        files: dict[str, bytes],
        overwrite_existing: bool = False,
    ) -> None:
        import mimetypes

        import lakefs
        import tqdm

        lk_branch: lakefs.Branch = (
            lakefs.repository(model.repo_id, client=self._client.lakefs_client)
            .branch(branch)
            .create(source_reference=model.default_branch, exist_ok=True)
        )

        # check for existing snapshot directory
        self._client.fs.source_branch = lk_branch.id
        model_dir = f"{model.repo_id}/{lk_branch.id}/"

        if self._client.fs.exists(model_dir) and not overwrite_existing:
            raise RuntimeError(
                f"Model snapshot '{model_dir}' already exists. "
                "Set overwrite_existing=True to overwrite."
            )

        for file_tgt, content in tqdm.tqdm(files.items(), desc="Uploading"):
            content_type = (
                mimetypes.guess_type(file_tgt)[0] or "application/octet-stream"
            )

            lk_branch.object(f"{file_tgt}").upload(content, content_type=content_type)

        self._commit_changes(
            repo_id=str(model.repo_id),
            branch=lk_branch.id,
            message="Upload model snapshot.",
        )

    def finalize(self, model: Model, branch: str) -> None:
        """Finalize a model in the hub."""

        from atriax_client.api.model import model_finalize

        with self._client.protected_api_client as client:
            response = model_finalize.sync_detailed(
                client=client, id=model.id, branch=branch
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to finalize model: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def upload_files(
        self,
        model: Model,
        branch: str,
        model_files: list[tuple[str, str]],
        overwrite_existing: bool = False,
    ) -> None:
        import mimetypes

        import lakefs
        import tqdm

        lk_branch: lakefs.Branch = (
            lakefs.repository(model.repo_id, client=self._client.lakefs_client)
            .branch(branch)
            .create(source_reference=model.default_branch, exist_ok=True)
        )

        # check for existing snapshot directory
        self._client.fs.source_branch = lk_branch.id
        model_dir = f"{model.repo_id}/{lk_branch.id}/"
        if self._client.fs.exists(model_dir) and not overwrite_existing:
            raise RuntimeError(
                f"Model snapshot '{model_dir}' already exists. "
                "Set overwrite_existing=True to overwrite."
            )

        for src, file_tgt in tqdm.tqdm(model_files, desc="Uploading"):
            content_type = mimetypes.guess_type(src)[0] or "application/octet-stream"
            with open(src, "rb") as f:
                lk_branch.object(file_tgt).upload(f.read(), content_type=content_type)

        self._commit_changes(
            repo_id=str(model.repo_id),
            branch=lk_branch.id,
            message="Upload model snapshot.",
        )

    def download_files(
        self, model_repo_id: str, branch: str, destination_path: str
    ) -> None:
        from pathlib import Path

        from fsspec.callbacks import TqdmCallback

        src = f"{model_repo_id}/{branch}/"
        tgt = str(Path(destination_path))
        self._client.fs.get(
            src,
            tgt,
            recursive=True,
            callback=TqdmCallback(tqdm_kwargs={"desc": "Downloading model"}),
        )

    def _commit_changes(self, repo_id: str, branch: str, message: str) -> None:
        import lakefs

        lk_branch = lakefs.repository(
            repo_id, client=self._client.lakefs_client
        ).branch(branch)
        uncommitted = list(lk_branch.uncommitted())
        if uncommitted:
            lk_branch.commit(message=message)
