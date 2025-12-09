from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from atria_metrics import MetricConfig
from atria_registry import ModuleConfig
from atria_registry._module_base import BaseModel
from atria_transforms.core._tfs._base import DataTransform
from pydantic import SerializeAsAny

from atria_models.core.model_builders._common import FrozenLayers, ModelBuilderType

if TYPE_CHECKING:
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

_DEFAULT_OPTIMIZER_PARAMETERS_KEY = "default"
_REQUIRED_DEFAULT = "???"


class ModelConfig(BaseModel):
    # builder type to use for model construction
    builder_type: ModelBuilderType = ModelBuilderType.timm
    bn_to_gn: bool = False
    frozen_layers: FrozenLayers | list[str] = FrozenLayers.none
    pretrained_checkpoint: str | None = None

    # Path or name of the model to build
    model_name_or_path: str = "resnet18"
    model_kwargs: dict[str, object] = {}


class ModelPipelineConfig(ModuleConfig):
    model: ModelConfig = ModelConfig()
    train_transform: SerializeAsAny[DataTransform] | None = (
        None  # the type at runtime can be child of DataTransform so we need to use SerializeAsAny
    )
    eval_transform: SerializeAsAny[DataTransform] | None = (
        None  # the type at runtime can be child of DataTransform so we need to use SerializeAsAny
    )
    metrics: list[MetricConfig] | None = None

    # @field_validator("metrics", mode="before"):
    # @classmethod
    # def _validate_metrics(
    #     cls, v: Any
    # ) -> list[MetricConfig] | None:
    #     if v is None:
    #         return None
    #     assert isinstance(v, list), "metrics must be a list"
    #     for item in v:
    #         assert isinstance(
    #             item, (dict, MetricConfig)
    #         ), "Each metric must be a dict or MetricConfig instance"
    #     return [MetricConfig.model_validate(item) for item in v]

    def build(self, **kwargs: Any) -> ModelPipeline:
        labels = kwargs.pop("labels")
        assert labels is not None, (
            "Labels must be provided to build the model pipeline."
        )
        return super().build(labels=labels, **kwargs)


T_ModelPipelineConfig = TypeVar("T_ModelPipelineConfig", bound=ModelPipelineConfig)
