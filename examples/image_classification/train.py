import fire
from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_ml.configs import (
    DataConfig,
    RuntimeEnvConfig,
    TrainerConfig,
    TrainingTaskConfig,
)
from atria_ml.optimizers._api import load_optimizer_config
from atria_ml.task_pipelines._trainer import Trainer
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform


def main(
    project_name: str = "my_atria_project",
    dataset_name: str = "cifar10/default",
    model_name: str = "resnet50",
    exp_name: str = "train_img_cls_00",
    output_dir: str = "./outputs",
    stats: str = "imagenet",
    image_size: int = 224,
    max_epochs: int = 40,
    train_batch_size: int = 64,
    eval_batch_size: int = 64,
    num_workers: int = 8,
    seed: int = 42,
    optim: str = "adamw",
    lr: float = 0.001,
):
    config = TrainingTaskConfig(
        env=RuntimeEnvConfig(
            project_name=project_name,
            exp_name=exp_name,
            dataset_name=dataset_name.replace("/", "_"),
            model_name=model_name,
            output_dir=output_dir,
            seed=seed,
        ),
        model_pipeline=load_model_pipeline_config(
            "image_classification",
            model=ModelConfig(model_name_or_path=model_name),
            train_transform=load_transform(
                "image_processor",
                tf={
                    "stats": stats,
                    "resize_height": image_size,
                    "resize_width": image_size,
                },
            ),
            eval_transform=load_transform(
                "image_processor",
                tf={
                    "stats": stats,
                    "resize_height": image_size,
                    "resize_width": image_size,
                },
            ),
        ),
        data=DataConfig(
            dataset_config=load_dataset_config("cifar10/default"),
            num_workers=num_workers,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        ),
        trainer=TrainerConfig(
            max_epochs=max_epochs,
            optimizer=load_optimizer_config(
                optimizer_name=optim,
                lr=lr,
            ),
        ),
        do_train=True,
        do_validation=True,
        do_test=True,
    )
    trainer = Trainer(config=config)
    trainer.run()


if __name__ == "__main__":
    fire.Fire(main)
