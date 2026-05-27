from __future__ import annotations

from typing import TYPE_CHECKING

from atria_hub.api.base import BaseApi
from atria_hub.utilities import get_logger

if TYPE_CHECKING:
    import uuid

    from atria_core.types.common import DatasetSplitType
    from atriax_client.models.data_instance_type import DataInstanceType
    from atriax_client.models.dataset import Dataset

    from atria_hub.api.base import BaseApi
    from atria_hub.utilities import get_logger

logger = get_logger(__name__)


class DatasetNotFoundError(Exception):
    """Exception raised when a dataset is not found in the hub."""

    def __init__(self, username: str, name: str):
        super().__init__(f"Dataset {username}/{name} not found in the hub.")
        self.username = username
        self.name = name


class DatasetsApi(BaseApi):
    def get(self, id: uuid.UUID) -> Dataset:
        """Retrieve a dataset from the hub by its name."""

        from atriax_client.api.dataset import dataset_item

        with self._client.protected_api_client as client:
            response = dataset_item.sync_detailed(id, client=client)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get dataset: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def get_by_name(self, username: str, name: str):
        """Retrieve a dataset from the hub by its name."""

        from atriax_client.api.dataset import dataset_find_one

        with self._client.protected_api_client as client:
            response = dataset_find_one.sync_detailed(
                client=client, username=username, name=name
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get dataset: {response.status_code} - {response.content.decode('utf-8')}"
                )
            if response.status_code == 404:
                raise DatasetNotFoundError(username=username, name=name)
            return response.parsed

    def create(
        self,
        name: str,
        default_branch: str = "main",
        description: str | None = None,
        data_instance_type: DataInstanceType | None = None,
        is_public: bool = False,
    ):
        """Create a new dataset in the hub."""

        from atriax_client.api.dataset import dataset_create
        from atriax_client.models.body_dataset_create import BodyDatasetCreate

        with self._client.protected_api_client as client:
            response = dataset_create.sync_detailed(
                client=client,
                body=BodyDatasetCreate(
                    name=name,
                    description=description,
                    data_instance_type=data_instance_type,
                    default_branch=default_branch,
                    is_public=is_public,
                ),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to create dataset: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def get_or_create(
        self,
        username: str,
        name: str,
        default_branch: str = "main",
        description: str | None = None,
        data_instance_type: DataInstanceType | None = None,
        is_public: bool = False,
    ) -> Dataset:
        """Get or create a dataset in the hub."""

        try:
            return self.get_by_name(username=username, name=name)
        except Exception:
            return self.create(
                name=name,
                default_branch=default_branch,
                description=description,
                data_instance_type=data_instance_type,
                is_public=is_public,
            )

    def upload_files(
        self,
        dataset: Dataset,
        branch: str,
        config_dir: str,
        dataset_files: list[tuple[str, str]],
        overwrite_existing: bool = False,
    ) -> None:
        import mimetypes

        import lakefs
        import tqdm

        branch: lakefs.Branch = (
            lakefs.repository(dataset.repo_id, client=self._client.lakefs_client)
            .branch(branch)
            .create(source_reference=dataset.default_branch, exist_ok=True)
        ).id

        # get target repository path
        self._client.fs.source_branch = branch
        tgt = f"{dataset.repo_id}/{branch}/"

        # first verify that delta directory already does not exist
        deltadir = f"{tgt}{config_dir}/delta/"
        if self._client.fs.exists(deltadir) and not overwrite_existing:
            raise RuntimeError(
                f"Delta directory {deltadir} already exists. "
                f"Either choose a different branch or set overwrite_existing=True to overwrite. "
                f"to overwrite the dataset."
            )

        # iterate over the dataset files and upload them to the hub
        logger.info(
            f"Uploading {len(dataset_files)} files to dataset {dataset.name} in repo/branch {dataset.name}/{branch}..."
        )
        # Log a preview of files to be uploaded
        from rich.pretty import pretty_repr

        logger.info(
            f"Files to be uploaded:\n{pretty_repr(dataset_files, max_length=4)}, dataset.repo_id={dataset.repo_id}, branch={branch}, config_dir={config_dir}"
        )
        for file in tqdm.tqdm(dataset_files, desc="Uploading"):
            # if it is a yaml file, we need to set the content type
            src, file_tgt = file

            content_type = mimetypes.guess_type(src)
            if content_type is None:
                content_type = "application/octet-stream"
            else:
                content_type = content_type[0]

            # this is slow but for now it works, in future we could use s3 gateway with async boto3 client instead
            self._client.fs.put_file(
                lpath=src,
                rpath=f"{tgt}{file_tgt}",
                precheck=False,
                content_type=content_type,
            )

    def download_files(
        self, dataset_repo_id: str, branch: str, config_dir: str, destination_path: str
    ) -> None:
        """Download files from a dataset."""
        from pathlib import Path

        from fsspec.callbacks import TqdmCallback

        src = f"{dataset_repo_id}/{branch}/{config_dir}/"
        tgt = str(Path(destination_path) / config_dir)
        self._client.fs.get(
            src,
            tgt,
            recursive=True,
            callback=TqdmCallback(tqdm_kwargs={"desc": "Downloading files"}),
        )

    def get_splits(
        self, dataset_repo_id: str, branch: str, config_name: str
    ) -> list[DatasetSplitType]:
        from pathlib import Path

        from atria_core.types.common import DatasetSplitType

        # get target repository path
        tgt = f"{dataset_repo_id}/{branch}/{config_name}/delta/"

        # first verify that delta directory already does not exist
        dir_ls = self._client.fs.ls(tgt)
        return [
            DatasetSplitType(Path(x["name"]).name)
            for x in dir_ls
            if Path(x["name"]).name in DatasetSplitType.__members__
        ]

    def get_available_configs(self, dataset_repo_id: str, branch: str) -> bool:
        """Check if a configuration exists in the dataset."""
        from pathlib import Path

        dir_ls = self._client.fs.ls(f"{dataset_repo_id}/{branch}/conf/dataset/")
        return [Path(x["name"]).name.replace(".yaml", "") for x in dir_ls]

    def get_config(self, dataset_repo_id: str, branch: str, config_name: str) -> dict:
        from pathlib import Path

        import lakefs
        import yaml

        # list all configs in the branch
        dir_ls = self._client.fs.ls(f"{dataset_repo_id}/{branch}/conf/dataset/")
        if not any(Path(x["name"]).name == f"{config_name}.yaml" for x in dir_ls):
            raise RuntimeError(
                f"Configuration '{config_name}' not found in the dataset on branch {branch}."
                f"Available configurations: {[Path(x['name']).name.replace('.yaml', '') for x in dir_ls]}"
            )
        branch: lakefs.Branch = lakefs.repository(
            dataset_repo_id, client=self._client.lakefs_client
        ).branch(branch)
        with branch.object(f"conf/dataset/{config_name}.yaml").reader(
            pre_sign=True
        ) as f:
            config = yaml.safe_load(f.read().decode("utf-8"))
        if not isinstance(config, dict):
            raise RuntimeError(
                f"The dataset configuration {config_name} is not a valid dictionary. "
            )
        return config

    def get_metadata(self, dataset_repo_id: str, branch: str) -> dict:
        import lakefs
        import yaml

        branch: lakefs.Branch = lakefs.repository(
            dataset_repo_id, client=self._client.lakefs_client
        ).branch(branch)
        with branch.object("metadata.yaml").reader(pre_sign=True) as f:
            config = yaml.safe_load(f.read().decode("utf-8"))
        if not isinstance(config, dict):
            raise ValueError(
                "The dataset metadata is not a valid dictionary. "
                "Please ensure the dataset was saved with the metadata."
            )
        return config

    def read_dataset_info(self, dataset_repo_id: str, branch: str) -> tuple[dict, dict]:
        """Read dataset info from the hub."""

        config = self.get_config(dataset_repo_id, branch)
        metadata = self.get_metadata(dataset_repo_id, branch)
        return config, metadata

    def commit_changes(self, dataset_repo_id: str, branch: str, message: str) -> None:
        """Commit changes to the dataset."""
        import lakefs
        from lakefs.branch import Branch

        branch: Branch = lakefs.repository(
            dataset_repo_id, client=self._client.lakefs_client
        ).branch(branch)
        uncommitted_changes = list(branch.uncommitted())
        if len(uncommitted_changes) == 0:
            return
        return branch.commit(message=message)

    def dataset_table_path(
        self, dataset_repo_id: str, branch: str, config_name: str, split: str
    ) -> str:
        return f"lakefs://{dataset_repo_id}/{branch}/{config_name}/delta/{split}/"

    def get_or_create_eval_branch(
        self, dataset_repo_id: str, dataset_branch: str
    ) -> str:
        import lakefs

        commit_sha = self.get_commit_sha(dataset_repo_id, dataset_branch)
        eval_branch = f"eval-{dataset_branch}-{commit_sha[:7]}"
        lakefs.repository(dataset_repo_id, client=self._client.lakefs_client).branch(
            eval_branch
        ).create(source_reference=dataset_branch, exist_ok=True)
        return eval_branch

    def eval_base_path(
        self,
        dataset_repo_id: str,
        eval_branch: str,
        config_name: str,
        split: str,
        output_path: str,
    ):
        return f"lakefs://{dataset_repo_id}/{eval_branch}/{config_name}/eval/{split}/{output_path}"

    def eval_table_path(
        self,
        dataset_repo_id: str,
        eval_branch: str,
        config_name: str,
        split: str,
        output_path: str,
    ) -> str:
        return (
            self.eval_base_path(
                dataset_repo_id=dataset_repo_id,
                eval_branch=eval_branch,
                split=split,
                config_name=config_name,
                output_path=output_path,
            )
            + "/delta"
        )

    def eval_metrics_path(
        self,
        dataset_repo_id: str,
        eval_branch: str,
        config_name: str,
        split: str,
        output_path: str,
    ) -> str:
        return (
            self.eval_base_path(
                dataset_repo_id=dataset_repo_id,
                eval_branch=eval_branch,
                split=split,
                config_name=config_name,
                output_path=output_path,
            )
            + "/metrics.json"
        )

    def write_eval_metrics(
        self,
        dataset_repo_id: str,
        eval_branch: str,
        config_name: str,
        split: str,
        output_path: str,
        data: dict,
    ) -> str:
        import json

        import lakefs
        from lakefs.branch import Branch

        eval_branch: Branch = lakefs.repository(
            dataset_repo_id, client=self._client.lakefs_client
        ).branch(eval_branch)
        eval_metrics_path = self.eval_metrics_path(
            dataset_repo_id=dataset_repo_id,
            eval_branch=eval_branch.id,
            config_name=config_name,
            split=split,
            output_path=output_path,
        )
        eval_branch.object(eval_metrics_path).upload(json.dumps(data).encode("utf-8"))
        eval_branch.commit(
            message=f"Write evaluation metrics for {dataset_repo_id} on {eval_branch.id} for split {split}",
            paths=[eval_metrics_path],
        )
        return eval_metrics_path

    def read_eval_metrics(
        self,
        dataset_repo_id: str,
        eval_branch: str,
        config_name: str,
        split: str,
        output_path: str,
    ) -> tuple[str, dict]:
        import lakefs
        from lakefs.branch import Branch

        eval_branch: Branch = lakefs.repository(
            dataset_repo_id, client=self._client.lakefs_client
        ).branch(eval_branch)

        eval_metrics_path = self.eval_metrics_path(
            dataset_repo_id=dataset_repo_id,
            eval_branch=eval_branch.id,
            split=split,
            config_name=config_name,
            output_path=output_path,
        )

        if not eval_branch.object(eval_metrics_path).exists():
            return None, {}

        with eval_branch.object(eval_metrics_path).reader(pre_sign=True) as f:
            return eval_metrics_path, f.read().decode("utf-8")

    def delete(self, dataset: Dataset) -> None:
        """Delete a dataset from the hub."""

        from atriax_client.api.dataset import dataset_delete

        with self._client.protected_api_client as client:
            response = dataset_delete.sync_detailed(client=client, id=dataset.id)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to delete dataset: {response.status_code} - {response.content.decode('utf-8')}"
                )
