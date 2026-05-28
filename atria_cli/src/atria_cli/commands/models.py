from atria_logger import get_logger

logger = get_logger(__name__)


def upload(
    name: str,
    ckpt_path: str,
    branch: str = "main",
    is_public: bool = False,
    overwrite_existing: bool = False,
):
    """
    Creates a safetensors snapshot from a ModelPipeline checkpoint and uploads it to Atria Hub.
    """
    try:
        import torch
        from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

        pipeline = torch.load(ckpt_path, map_location="cpu")
        assert isinstance(pipeline, ModelPipeline), (
            f"Expected a ModelPipeline instance, got {type(pipeline)}"
        )
        pipeline.upload_to_hub(
            name=name,
            branch=branch,
            is_public=is_public,
            overwrite_existing=overwrite_existing,
        )
    except Exception as e:
        logger.exception("Failed to upload model:", exc_info=e)


def download(
    name: str,
    config_name: str = "default",
    branch: str = "main",
    download_dir: str | None = None,
):
    """
    Downloads a model snapshot from Atria Hub and loads it as a ModelPipeline.
    """
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

    ModelPipeline.load_from_hub(
        name=name, branch=branch, config_name=config_name, download_dir=download_dir
    )
