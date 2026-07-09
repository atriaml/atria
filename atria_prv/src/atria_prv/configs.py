from typing import Self

from atria_datasets.core.dataset_splitters._standard_splitter import StandardSplitter
from atria_ml.configs._data import DataConfig
from atria_ml.configs._task import TrainingTaskConfig
from atria_types._common import DatasetSplitType
from pydantic import BaseModel, Field


class DPConfig(BaseModel):
    noise_multiplier: float | None = None
    target_epsilon: float = 5.0
    target_delta: float | None = None  # 1 / len (dataset)
    max_grad_norm: float = 1.0
    sample_rate: float = Field(
        default=0.01,
        description=(
            "Poisson subsampling rate on the original (full) data size. Each client's "
            "DP batch size is ceil(len(local_shard) * sample_rate); also used as the DP "
            "accounting sample rate q. All clients participate every round, so "
            "client_fraction plays no role in privacy."
        ),
    )
    loss_reduction: str = "mean"
    clipping: str = "flat"
    accountant: str = "rdp"
    global_clip: float | None = None
    max_physical_batch_size: int = 8
    use_bmm: bool = True


class DPTrainingTaskConfig(TrainingTaskConfig):
    dp_config: DPConfig


class FLConfig(BaseModel):
    num_rounds: int = Field(default=10, description="Maximum number of rounds.")
    total_num_clients: int = Field(
        default=2, description="Total number of clients that are present."
    )
    client_fraction: float = Field(
        default=1.0,
        description="Fraction of total_num_clients sampled to participate in each round (0 < f <= 1).",
    )
    partition_type: str = "equal"
    seed: int = Field(
        default=42,
        description="Seed for dataset partitioning and per-round client sampling (independent of env.seed).",
    )
    validate_every_n_rounds: int = Field(
        default=1,
        description="Run global validation every N rounds, for tracking only.",
    )


class ClientDataConfig(DataConfig):
    partition_id: int | None = None
    partition_cache_dir: str | None = None

    @classmethod
    def from_data_config(
        cls,
        data_config: DataConfig,
        partition_id: int | None,
        partition_cache_dir: str | None,
    ) -> Self:
        return cls(
            dataset_config=data_config.dataset_config,
            data_dir=data_config.data_dir,
            access_token=data_config.access_token,
            overwrite_existing_cached=data_config.overwrite_existing_cached,
            allowed_keys=data_config.allowed_keys,
            num_processes=data_config.num_processes,
            cached_storage_type=data_config.cached_storage_type,
            enable_cached_splits=data_config.enable_cached_splits,
            store_artifact_content=data_config.store_artifact_content,
            max_cache_image_size=data_config.max_cache_image_size,
            train_batch_size=data_config.train_batch_size,
            eval_batch_size=data_config.eval_batch_size,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            splitting_enabled=data_config.splitting_enabled,
            split_ratio=data_config.split_ratio,
            preprocess_train_transform=data_config.preprocess_train_transform,
            preprocess_eval_transform=data_config.preprocess_eval_transform,
            preprocess_max_cache_image_size=data_config.preprocess_max_cache_image_size,
            partition_id=partition_id,
            partition_cache_dir=partition_cache_dir,
        )

    def build_dataset(
        self,
    ) -> Self:
        dataset = self.dataset_config.build(
            data_dir=self.data_dir,
            access_token=self.access_token,
            overwrite_existing_cached=self.overwrite_existing_cached,
            allowed_keys=self.allowed_keys,
            num_processes=self.num_processes,
            cached_storage_type=self.cached_storage_type,
            enable_cached_splits=self.enable_cached_splits,
            store_artifact_content=self.store_artifact_content,
            max_cache_image_size=self.preprocess_max_cache_image_size,
            preprocess_train_transform=self.preprocess_train_transform,
            preprocess_eval_transform=self.preprocess_eval_transform,
            partition_id=self.partition_id,
            partition_cache_dir=self.partition_cache_dir,
        )
        if (
            DatasetSplitType.validation not in dataset.split_iterators
            and self.splitting_enabled
        ):
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


class FLTrainingTaskConfig(TrainingTaskConfig):
    fl_config: FLConfig


class ServerAdamConfig(BaseModel):
    lr: float = Field(
        default=1e-2, description="Server-side global Adam learning rate (eta)."
    )
    beta1: float = Field(default=0.9, description="Adam beta1.")
    beta2: float = Field(default=0.99, description="Adam beta2.")
    eps: float = Field(default=1e-3, description="Adam epsilon (tau).")


class FeAmDPTrainingTaskConfig(TrainingTaskConfig):
    fl_config: FLConfig
    dp_config: DPConfig
    server_optim: ServerAdamConfig = ServerAdamConfig()
