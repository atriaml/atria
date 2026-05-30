from __future__ import annotations

from atria_datasets.core.dataset._cached_dataset import CachedDataset
from atria_logger import get_logger

logger = get_logger(__name__)


def prepare_and_upload(
    name: str,
    target_name: str | None = None,
    branch: str = "main",
    is_public: bool = False,
    overwrite_existing: bool = False,
    data_dir: str | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    num_processes: int = 8,
    enable_cached_splits: bool = True,
    store_artifact_content: bool = True,
    max_cache_image_size: int | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_test_samples: int | None = None,
):
    """
    Uploads a dataset to the Atria Hub.
    """
    try:
        from atria_datasets import FileStorageType, load_dataset_config

        dataset_config = load_dataset_config(
            f"{name}",
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            max_test_samples=max_test_samples,
        )
        dataset = dataset_config.build(
            data_dir=data_dir,
            access_token=access_token,
            overwrite_existing_cached=overwrite_existing_cached,
            num_processes=num_processes,
            cached_storage_type=FileStorageType.DELTALAKE,
            enable_cached_splits=enable_cached_splits,
            store_artifact_content=store_artifact_content,
            max_cache_image_size=max_cache_image_size,
        )
        logger.info(f"Preparing dataset {name} for upload to Atria Hub...")
        assert isinstance(dataset, CachedDataset), (
            "Expected dataset to be a CachedDataset after preparation."
        )
        repo_info = dataset.upload_to_hub(
            name=target_name
            or dataset_config.dataset_name.replace("/", "-").replace("_", "-"),
            branch=branch,
            is_public=is_public,
            overwrite_existing=overwrite_existing,
        )
        repo_path = f"{repo_info['username']}/{repo_info['name']}@{repo_info['branch']}"
        logger.info(f"Dataset uploaded to path: {repo_path}")
    except Exception as e:
        logger.exception(e)


def download(name: str, branch: str = "main", download_dir: str | None = None):
    """
    Downloads a dataset from the Atria Hub.
    """

    logger.info(f"Downloading dataset {name} from Atria Hub...")
    dataset = CachedDataset.load_from_hub(
        name=name, branch=branch, download_dir=download_dir
    )

    logger.info(f"dataset loaded from hub successfully: \n{dataset}")
