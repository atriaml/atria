from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform

from atria_insights.configs.explainer_config import (
    DataConfig,
    ExplainerRunConfig,
    RuntimeEnvConfig,
)
from atria_insights.task_pipelines.explainer import ModelExplainer

config = ExplainerRunConfig(
    env=RuntimeEnvConfig(
        project_name="my_atria_project",
        run_name="cifar10_experiment_03",
        output_dir="./outputs/",
        seed=42,
    ),
    model_pipeline=load_model_pipeline_config(
        "image_classification",
        model=ModelConfig(model_name_or_path="resnet50"),
        train_transform=load_transform(
            "image_processor", tf={"use_imagenet_mean_std": True}
        ),
        eval_transform=load_transform(
            "image_processor", tf={"use_imagenet_mean_std": True}
        ),
    ),
    data=DataConfig(
        dataset_config=load_dataset_config("cifar10/default"),
        data_dir="data_dir/",
        num_workers=0,
        train_batch_size=64,
        eval_batch_size=64,
    ),
)
explanation_service = ModelExplainer(config=config)
explanation_service.run()
