import fire
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401

from atria_insights import ExplanationTaskConfig, ModelExplainer


def main(
    checkpoint_path: str,
    dataset_name: str | None = None,
    exp_name: str = "explain_img_cls_00",
    output_dir: str = "./outputs",
):
    import torch
    from atria_ml.configs import TrainingTaskConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TrainingTaskConfig.from_dict(checkpoint["config"])
    config = ExplanationTaskConfig.from_training_task_config(
        training_task_config=config,
        dataset_name=dataset_name,
        exp_name=exp_name,
        output_dir=output_dir,
    )
    model_explainer = ModelExplainer(config=config)
    state = model_explainer.run(checkpoint_path=checkpoint_path, total_samples=10)
    print(state.output)


if __name__ == "__main__":
    fire.Fire(main)
