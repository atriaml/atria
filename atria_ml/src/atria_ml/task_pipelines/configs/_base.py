from __future__ import annotations

import hashlib
import json
from pathlib import Path

import codename
from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.core.dataset._common import DatasetConfig
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset_splitters.standard_splitter import StandardSplitter
from atria_datasets.core.storage.utilities import FileStorageType
from atria_logger import get_logger
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from atria_registry._module_base import BaseModel
from atria_types._utilities._repr import RepresentationMixin
from pydantic import ConfigDict, Field

from atria_ml.optimizers._base import OptimizerConfig
from atria_ml.optimizers._torch import SGDOptimizerConfig
from atria_ml.schedulers._base import LRSchedulerConfig
from atria_ml.schedulers._torch import CosineAnnealingLRSchedulerConfig
from atria_ml.training._configs import (
    EarlyStoppingConfig,
    GradientConfig,
    LoggingConfig,
    ModelCheckpointConfig,
    ModelEmaConfig,
    WarmupConfig,
)

logger = get_logger(__name__)


class RuntimeEnvConfig(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    project_name: str = "atria_ml"
    run_name: str = Field(default_factory=lambda: codename.codename())
    output_dir: str = "???"
    seed: int = 42
    deterministic: bool = False
    backend: str | None = "nccl"
    n_devices: int = 1


class DataConfig(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
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


class TrainerConfig(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    max_epochs: int = 10
    validate_every_n_epochs: float = 1.0
    visualize_every_n_epochs: float = 1.0
    resume_checkpoint_path: str | None = None
    clear_cuda_cache: bool = True
    stop_on_nan: bool = True
    eval_training: bool = False
    outputs_to_running_avg: list[str] = Field(default_factory=lambda: ["loss"])

    optimizer: dict[str, OptimizerConfig] | OptimizerConfig = SGDOptimizerConfig(
        lr=1e-3, momentum=0.9, weight_decay=0.0
    )
    lr_scheduler: dict[str, LRSchedulerConfig] | LRSchedulerConfig | None = (
        CosineAnnealingLRSchedulerConfig()
    )
    model_ema: ModelEmaConfig = ModelEmaConfig()
    warmup: WarmupConfig = WarmupConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()
    gradient: GradientConfig = GradientConfig()


def pydantic_to_hydra(obj: BaseModel):
    """
    Recursively convert a Pydantic BaseModel into a dict suitable
    for Hydra instantiate (with _target_).
    """
    # First pass: collect all _target_ paths
    targets = {}

    def collect_targets(current_obj, path=""):
        if isinstance(current_obj, BaseModel):
            target = (
                f"{current_obj.__class__.__module__}.{current_obj.__class__.__name__}"
            )
            targets[path] = target

            # Traverse fields
            for field_name, field_value in current_obj.__dict__.items():
                field_path = f"{path}.{field_name}" if path else field_name
                collect_targets(field_value, field_path)

        elif isinstance(current_obj, list):
            for i, item in enumerate(current_obj):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                collect_targets(item, item_path)

        elif isinstance(current_obj, dict):
            for key, value in current_obj.items():
                key_path = f"{path}.{key}" if path else key
                collect_targets(value, key_path)

    # Collect all target paths
    collect_targets(obj)

    # Second pass: dump model and assign targets
    data = obj.model_dump()

    def assign_targets(current_data, path=""):
        if path in targets:
            if isinstance(current_data, dict):
                current_data["_target_"] = targets[path]

        if isinstance(current_data, dict):
            for key, value in current_data.items():
                key_path = f"{path}.{key}" if path else key
                assign_targets(value, key_path)
        elif isinstance(current_data, list):
            for i, item in enumerate(current_data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                assign_targets(item, item_path)

    # Assign targets to the dumped data
    assign_targets(data)

    return data


class RunConfig(RepresentationMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    env: RuntimeEnvConfig = Field(default_factory=RuntimeEnvConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model_pipeline: ModelPipelineConfig
    trainer: TrainerConfig = TrainerConfig()
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    test_run: bool = False
    use_fixed_batch_iterator: bool = False
    save_test_outputs_to_disk: bool = False
    use_ema_for_evaluation: bool = False
    with_amp: bool = True
    do_train: bool = True
    do_test: bool = True
    do_validation: bool = True
    do_visualization: bool = False
    reevaluate_metrics: bool = True

    def build_dataset(self) -> Dataset:
        return self.data.build_dataset()

    def state_dict(self) -> dict:
        return self.model_dump()

    def load_state_dict(self, state_dict: dict) -> None:
        self.model_validate(state_dict)

    def get_metrics_file_path(self) -> Path:
        params = self.model_dump()
        config_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:8]
        return Path(self.env.output_dir) / f"outputs-{config_hash}.json"

    def metrics_file_exists(self) -> bool:
        output_file_path = self.get_metrics_file_path()
        return output_file_path.exists()

    def dump_metrics_file(self, data: dict) -> None:
        output_file_path = self.get_metrics_file_path()
        with open(output_file_path, "w") as f:
            json.dump({"config": self.model_dump(), "data": data}, f, indent=4)
        logger.info(f"Metrics dumped to {output_file_path}")

    def save_to_json(self, file_path: str | Path | None = None) -> None:
        if file_path is None:
            file_path = Path(self.env.output_dir) / "config.json"
        else:
            file_path = Path(file_path)

        # make parent
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use pydantic_to_hydra to convert to Hydra-compatible format
        hydra_data = pydantic_to_hydra(self)

        with open(file_path, "w") as f:
            json.dump(hydra_data, f, indent=4)

        logger.info(f"RunConfig saved to {file_path}")

    @classmethod
    def from_json(cls, file_path: str | Path) -> RunConfig:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        file_path = Path(file_path)
        with open(file_path) as f:
            data = json.load(f)

        # Convert to OmegaConf
        omega_conf = OmegaConf.create(data)

        # Use Hydra instantiate to create the object
        return instantiate(omega_conf)


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
