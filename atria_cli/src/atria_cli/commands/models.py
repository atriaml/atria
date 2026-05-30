from pathlib import Path

from atria_logger import get_logger

logger = get_logger(__name__)


def upload(
    name: str,
    snapshot_dir: str,
    branch: str = "main",
    is_public: bool = False,
    overwrite_existing: bool = False,
):
    """
    Creates a safetensors snapshot from a ModelPipeline checkpoint and uploads it to Atria Hub.
    """
    try:
        from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

        assert Path(snapshot_dir).exists(), (
            f"snapshot_dir '{snapshot_dir}' does not exist"
        )
        loaded_pipeline = ModelPipeline.load_from_snapshot(Path(snapshot_dir))

        # upload to hub
        upload_result = loaded_pipeline.upload_to_hub(
            name=name,
            branch=branch,
            is_public=is_public,
            overwrite_existing=overwrite_existing,
        )
        logger.info(f"Upload result: {upload_result}")
    except Exception as e:
        logger.exception("Failed to upload model:", exc_info=e)


def download(name: str, branch: str = "main", download_dir: str | None = None):
    """
    Downloads a model snapshot from Atria Hub and loads it as a ModelPipeline.
    """
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

    ModelPipeline.load_from_hub(name=name, branch=branch, download_dir=download_dir)
