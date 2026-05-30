"""Test save_snapshot / _load_from_snapshot / upload_to_hub / load_from_hub for ModelPipeline."""

import torch
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_builders._common import ModelBuilderType
from atria_models.core.model_pipelines._common import ModelConfig
from atria_models.core.model_pipelines._hub_ops import ModelHubOps
from atria_types import DatasetLabels


def _weights_match(a, b) -> bool:
    sd_a = a._model.state_dict()
    sd_b = b._model.state_dict()
    if sd_a.keys() != sd_b.keys():
        return False
    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


def main(upload: bool = False):
    labels = DatasetLabels(classification=["cat", "dog", "bird"])

    model_pipeline_config = load_model_pipeline_config(
        "image_classification",
        model=ModelConfig(
            model_name_or_path="resnet18", builder_type=ModelBuilderType.timm
        ),
    )
    pipeline = model_pipeline_config.build(labels=labels)
    print("Built pipeline:", type(pipeline).__name__)

    # --- local round-trip ---
    snapshot_dir = pipeline.save_snapshot()
    print("Snapshot saved to:", snapshot_dir)
    print("Files:", [f.name for f in snapshot_dir.iterdir()])

    if not upload:
        reloaded = ModelHubOps.load_from_snapshot(snapshot_dir)
        assert _weights_match(pipeline, reloaded), "Local round-trip: weight mismatch!"
        print("Local round-trip OK.")

        return

    # --- hub upload ---
    repo_info = pipeline.upload_to_hub(
        name="resnet18-image-classification-test", overwrite_existing=True
    )
    print("Uploaded to hub:", repo_info)

    # --- hub download ---
    import tempfile

    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

    with tempfile.TemporaryDirectory() as tmp:
        hub_pipeline = ModelPipeline.load_from_hub(
            name=f"{repo_info['username']}/{repo_info['name']}",
            branch=repo_info["branch"],
            download_dir=tmp,
        )
        print("hub_pipeline,", hub_pipeline)
        assert _weights_match(pipeline, hub_pipeline), (
            "Hub round-trip: weight mismatch!"
        )
        print("Hub round-trip OK.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
