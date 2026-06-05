from pathlib import Path
from typing import Literal

from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_ml.configs import (
    DataConfig,
    RuntimeEnvConfig,
    TrainerConfig,
    TrainingTaskConfig,
    WarmupConfig,
)
from atria_ml.configs._task import EvaluationTaskConfig
from atria_ml.optimizers._api import load_optimizer_config
from atria_ml.task_pipelines._evaluator import Evaluator
from atria_ml.training._configs import EarlyStoppingConfig, ModelCheckpointConfig
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_builders._common import ModelBuilderType

# load the model from snapshot to verify it works
from atria_models.core.model_pipelines import ModelPipeline
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform
from atria_transforms.tfs._image_transforms import StandardImageTransform


def main(
    project_name: str = "my_atria_project",
    dataset_name: str = "tobacco3482/image_with_ocr",
    model_name: str = "bert-base-uncased",
    tokenizer_name: str = "bert-base-uncased",
    builder_type: ModelBuilderType = ModelBuilderType.atria,
    exp_name: str = "train_seq_cls_01",
    output_dir: str = "./outputs",
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "standard",
    image_size: int = 224,
    max_epochs: int = 100,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_workers: int = 8,
    seed: int = 42,
    optim: str = "adam",
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    splitting_enabled: bool = True,
    split_ratio: float = 0.95,
    run_eval: bool = False,
    eval_checkpoint: str | None = None,
    upload_name: str = "my_uploaded_model",
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
            "sequence_classification",
            model=ModelConfig(
                model_name_or_path=model_name,
                builder_type=builder_type,
                model_type="sequence_classification",
            ),
            train_transform=load_transform(
                "document_processor/sequence_classification",
                hf_processor={
                    "tokenizer_name": tokenizer_name,
                },
                image_transform=StandardImageTransform(
                    stats=stats, resize_width=image_size, resize_height=image_size
                ),
                overflow_strategy="return_first",
            ),
            eval_transform=load_transform(
                "document_processor/sequence_classification",
                hf_processor={
                    "tokenizer_name": tokenizer_name,
                },
                image_transform=StandardImageTransform(
                    stats=stats, resize_width=image_size, resize_height=image_size
                ),
                overflow_strategy="return_first",
            ),
        ),
        data=DataConfig(
            dataset_config=load_dataset_config(dataset_name),
            num_workers=num_workers,
            num_processes=num_workers,
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
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                monitored_metric="validation/accuracy",
                patience=10,
                mode="max",
            ),
            model_checkpoint=ModelCheckpointConfig(
                monitored_metric="validation/accuracy",
                mode="max",
            ),
        ),
        do_train=True,
        do_validation=True,
        do_test=True,
    )

    snapshot_dir = Path(config.env.run_dir) / "model_snapshot"
    if run_eval or not snapshot_dir.exists():
        evaluator = Evaluator(
            config=EvaluationTaskConfig.from_training_config(
                training_config=config, eval_checkpoint=eval_checkpoint
            )
        )

        if run_eval:
            evaluator.run()

        model_pipeline = evaluator._state.model_pipeline

        # save snapshot locally
        model_pipeline.save_to_disk(snapshot_dir=snapshot_dir)

    loaded_pipeline = ModelPipeline.load_from_disk(
        Path(config.env.run_dir) / "model_snapshot"
    )

    # upload to hub
    upload_result = loaded_pipeline.upload_to_hub(
        name=upload_name,
        branch="main",
        is_public=False,
        overwrite_existing=True,
    )
    print("Upload result:", upload_result)


if __name__ == "__main__":
    BASE_DIR = "/media/gladius/noel/phd-2026/atriax_project/docxeval_data/experiment_00_seq_cls_v3/"
    CONFIGS = [
        {
            "dataset_name": "tobacco3482/image_with_ocr",
            "model_name": "bert-base-uncased",
            "tokenizer_name": "bert-base-uncased",
            "eval_checkpoint": f"{BASE_DIR}/tobacco3482_image_with_ocr/bert-base-uncased/checkpoints/best_checkpoint_11_validation-accuracy=0.8638.pt",
            "upload_name": "tobacco3482-bert-base-uncased2",
        },
        {
            "dataset_name": "tobacco3482/image_with_ocr",
            "model_name": "layoutlmv3-base",
            "tokenizer_name": "microsoft/layoutlmv3-base",
            "eval_checkpoint": f"{BASE_DIR}/tobacco3482_image_with_ocr/layoutlmv3-base/checkpoints/best_checkpoint_11_validation-accuracy=0.9427.pt",
            "upload_name": "tobacco3482-layoutlmv3-base2",
        },
    ]
    for config in CONFIGS:
        main(
            project_name="atriax_demo",
            exp_name=f"eval_{config['dataset_name'].replace('/', '_')}_{config['model_name']}",
            dataset_name=config["dataset_name"],
            model_name=config["model_name"],
            eval_checkpoint=config["eval_checkpoint"],
            upload_name=config["upload_name"],
        )
