from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_ml.configs import (
    DataConfig,
    RuntimeEnvConfig,
    TrainerConfig,
    TrainingTaskConfig,
)
from atria_ml.task_pipelines._trainer import Trainer
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform

config = TrainingTaskConfig(
    env=RuntimeEnvConfig(
        project_name="my_atria_project",
        exp_name="train_00",
        dataset_name="cifar10",
        model_name="resnet50",
        output_dir="./outputs/",
        seed=42,
    ),
    model_pipeline=load_model_pipeline_config(
        "image_classification",
        model=ModelConfig(model_name_or_path="resnet50"),
        train_transform=load_transform("image_processor", tf={"stats": "imagenet"}),
        eval_transform=load_transform("image_processor", tf={"stats": "imagenet"}),
    ),
    data=DataConfig(
        dataset_config=load_dataset_config("cifar10/default"),
        num_workers=0,
        train_batch_size=64,
        eval_batch_size=64,
    ),
    trainer=TrainerConfig(),
    do_train=True,
    do_validation=True,
    do_test=True,
)
trainer = Trainer(config=config)
trainer.run()
