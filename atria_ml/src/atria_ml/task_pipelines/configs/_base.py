from __future__ import annotations

import codename
from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.core.dataset._common import DatasetConfig
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset_splitters.standard_splitter import StandardSplitter
from atria_datasets.core.storage.utilities import FileStorageType
from atria_logger import get_logger
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class RuntimeEnvConfig(BaseModel):
    project_name: str = "atria_ml"
    run_name: str = Field(default_factory=lambda: codename.codename())
    output_dir: str = "???"
    seed: int = 42
    deterministic: bool = False
    backend: str | None = "nccl"
    n_devices: int = 1


class DataConfig(RepresentationMixin, BaseModel):
    dataset_config: DatasetConfig = Field(
        default_factory=lambda: load_dataset_config("cifar10/1k")
    )
    data_dir: str | None = None

    # caching config
    access_token: str | None = None
    overwrite_existing_cached: bool = False
    allowed_keys: set[str] | None = None
    num_processes: int = 8
    cached_storage_type: FileStorageType = FileStorageType.MSGPACK
    enable_cached_splits: bool = False
    store_artifact_content: bool = True
    max_cache_image_size: int | None = None

    # dataloader config
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    # dataset split args
    splitting_enabled: bool = False
    split_ratio: float = 0.9

    def build_dataset(self) -> Dataset:
        dataset = self.dataset_config.build(
            data_dir=self.data_dir,
            access_token=self.access_token,
            overwrite_existing_cached=self.overwrite_existing_cached,
            allowed_keys=self.allowed_keys,
            num_processes=self.num_processes,
            cached_storage_type=self.cached_storage_type,
            enable_cached_splits=self.enable_cached_splits,
            store_artifact_content=self.store_artifact_content,
            max_cache_image_size=self.max_cache_image_size,
        )

        if dataset.validation is None and self.splitting_enabled:
            assert dataset.train is not None, (
                "Dataset splitting enabled but no training dataset found."
            )
            dataset_splitter = StandardSplitter(
                split_ratio=self.split_ratio, shuffle=True
            )
            train, validation = dataset_splitter(dataset.train)
            dataset.train = train
            dataset.validation = validation

            assert dataset.train is not None, (
                "Training split is None in the loaded dataset"
            )  # for our experiments we always make sure we have validation split present
            assert dataset.validation is not None, (
                "Validation split is None in the loaded dataset"
            )  # for our experiments we always make sure we have validation split present

        return dataset


class TrainerConfig(BaseModel):
    do_train: bool = True
    do_eval: bool = True
    do_validation: bool = True
    # evaluator configs
    with_amp: bool = False
    use_best: bool = True
    do_validation: bool = True
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    momentum: float = 0.9
    num_epochs: int = 50
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    enable_grad_clipping: bool = False
    max_grad_norm: float = 1.0
    # lr scheduler args
    lr_start: float = 1e-5
    lr_end: float = 1e-8
    lr_schedule_warmup_steps_frac_of_total: float = 0.1
    # validation args
    validate_every_n_epochs: float = 1.0
    enable_early_stopping: bool = True
    # checkpoint args
    save_ckpt_every_n_epochs: int = 1
    keep_n_checkpoints: int = 1
    monitored_metric: str | None = None
    monitored_metric_mode: str = "max"  # "min" or "max"
    save_weights_only: bool = False
    test_run: bool = False


class RunnerConfig(BaseModel):
    env: RuntimeEnvConfig = Field(default_factory=RuntimeEnvConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model_pipeline: ModelPipelineConfig
    trainer: TrainerConfig = TrainerConfig()

    def build_dataset(self) -> Dataset:
        return self.data.build_dataset()

    def state_dict(self) -> dict:
        return self.model_dump()

    def load_state_dict(self, state_dict: dict) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)


# class ImageClassificationTrainerConfig(RunnerConfig):
#     model_pipeline: ImageModelPipelineConfig
#     trainer: TrainerConfig = TrainerConfig()

#     def build_model_pipeline(self, labels: DatasetLabels) -> ImageModelPipeline:
#         model_pipeline = load_model_pipeline_config(
#             "image_classification",
#             model=self.model_pipeline.model,
#             mixup_config=self.model_pipeline.mixup_config,
#             train_transform=self.model_pipeline.train_transform,
#             eval_transform=self.model_pipeline.eval_transform,
#         )
#         return model_pipeline.build(labels=labels)
