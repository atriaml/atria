from typing import Self

from atria_ml.configs._data import DataConfig
from atria_ml.configs._task import TrainingTaskConfig
from pydantic import BaseModel, Field


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


class FLClientDataConfig(DataConfig):
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
            **{field: getattr(data_config, field) for field in DataConfig.model_fields},
            partition_id=partition_id,
            partition_cache_dir=partition_cache_dir,
        )

    def _build_kwargs(self) -> dict:
        kwargs = super()._build_kwargs()
        kwargs["partition_id"] = self.partition_id
        kwargs["partition_cache_dir"] = self.partition_cache_dir
        return kwargs


class FLTrainingTaskConfig(TrainingTaskConfig):
    fl_config: FLConfig


class FLClientTrainingTaskConfig(TrainingTaskConfig):
    client_id: int
    partition_cache_dir: str

    @classmethod
    def from_training_task_config(
        cls,
        training_task_config: TrainingTaskConfig,
        data_config: FLClientDataConfig,
        client_id: int,
        partition_cache_dir: str,
    ) -> Self:
        return cls(
            **{
                field: getattr(training_task_config, field)
                for field in TrainingTaskConfig.model_fields
                if field != "data"
            },
            data=data_config,
            client_id=client_id,
            partition_cache_dir=partition_cache_dir,
        )
