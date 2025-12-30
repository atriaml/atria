from typing import Literal

import fire
from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_ml.configs import (
    DataConfig,
    RuntimeEnvConfig,
    TrainerConfig,
    TrainingTaskConfig,
    WarmupConfig,
)
from atria_ml.optimizers._api import load_optimizer_config
from atria_ml.task_pipelines._trainer import Trainer
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_builders._common import ModelBuilderType
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform
from atria_transforms.tfs._image_transforms import StandardImageTransform


def main(
    project_name: str = "my_atria_project",
    dataset_name: str = "funsd",
    model_name: str = "lilt-roberta-base",
    builder_type: ModelBuilderType = ModelBuilderType.atria,
    exp_name: str = "train_layout_token_cls_03",
    output_dir: str = "./outputs",
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "standard",
    image_size: int = 224,
    max_epochs: int = 100,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 8,
    seed: int = 42,
    optim: str = "adamw",
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    overwrite_output_dir: bool = False,
    splitting_enabled: bool = True,
    split_ratio: float = 0.95,
):
    config = TrainingTaskConfig(
        env=RuntimeEnvConfig(
            project_name=project_name,
            exp_name=exp_name,
            dataset_name=dataset_name.replace("/", "_"),
            model_name=model_name,
            output_dir=output_dir,
            seed=seed,
            overwrite_output_dir=overwrite_output_dir,
        ),
        model_pipeline=load_model_pipeline_config(
            "layout_token_classification",
            model=ModelConfig(
                model_name_or_path=model_name,
                builder_type=builder_type,
                model_type="token_classification",
            ),
            train_transform=load_transform(
                "token_classification_document_processor",
                hf_processor={
                    "tokenizer_name": "bert-base-uncased",
                },
                image_transform=StandardImageTransform(
                    stats=stats, resize_width=image_size, resize_height=image_size
                ),
                overflow_strategy="return_random",
            ),
            eval_transform=load_transform(
                "token_classification_document_processor",
                hf_processor={
                    "tokenizer_name": "bert-base-uncased",
                },
                image_transform=StandardImageTransform(
                    stats=stats, resize_width=image_size, resize_height=image_size
                ),
                overflow_strategy="return_all",
            ),
        ),
        data=DataConfig(
            dataset_config=load_dataset_config(dataset_name),
            num_workers=num_workers,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            splitting_enabled=splitting_enabled,
            split_ratio=split_ratio,
        ),
        trainer=TrainerConfig(
            max_epochs=max_epochs,
            optimizer=load_optimizer_config(
                optimizer_name=optim,
                lr=lr,
                weight_decay=weight_decay,
            ),
            warmup=WarmupConfig(
                warmup_steps=warmup_steps,
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
