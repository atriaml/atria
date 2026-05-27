from atria_core.logger import get_logger
from atria_registry.utilities import instantiate_object_from_config

logger = get_logger(__name__)


def upload(
    name: str,
    ckpt_path: str,
    branch: str = "main",
    is_public: bool = False,
    description: str = None,
    overwrite_existing: bool = False,
):
    """
    Uploads a model to the Atria Hub and verifies configuration integrity.
    """
    try:
        import logging

        import deepdiff
        import torch
        from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline

        logger = logging.getLogger(__name__)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Extract and build model pipeline
        build_config = checkpoint["model_pipeline"]["atria_config"]
        model_pipeline: AtriaModelPipeline = instantiate_object_from_config(
            build_config
        ).build_from_checkpoint(checkpoint=checkpoint["model_pipeline"])

        # Make sure that the model pipeline is built correctly and matches the checkpoint
        diff = deepdiff.DeepDiff(
            build_config, model_pipeline.build_config, ignore_order=True
        )
        if diff:
            # this should not happen, the newly created build config must match the checkpoint
            # build config with which it is instantiated
            raise RuntimeError(
                f"Configuration mismatch: {diff}. The checkpoint model pipeline config does not match the "
                "instantiated model pipeline config. Please check the model pipeline configuration."
            )

        logger.info("Model pipeline instantiated successfully: %s", model_pipeline)
        model_pipeline.upload_to_hub(
            name=name,
            branch=branch,
            is_public=is_public,
            description=description,
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
    Downloads a model from the Atria Hub.
    """
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline

    AtriaModelPipeline.download_from_hub(
        name=name, branch=branch, config_name=config_name, download_dir=download_dir
    )
