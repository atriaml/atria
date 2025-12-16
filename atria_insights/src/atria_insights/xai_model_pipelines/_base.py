from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import TYPE_CHECKING, Generic, Literal

from atria_logger import get_logger
from atria_models.core.model_pipelines._common import T_ModelPipelineConfig
from atria_models.core.types.model_outputs import ModelOutput
from atria_registry._module_base import ConfigurableModule
from atria_transforms.core._data_types._base import TensorDataModel
from atria_types._datasets import DatasetLabels

if TYPE_CHECKING:
    import torch
    from atria_models.core.model_pipelines._ops import ModelPipelineOps
    from ignite.engine import Engine
    from ignite.metrics import Metric

logger = get_logger(__name__)


class XAIModelPipeline(
    ConfigurableModule[T_ModelPipelineConfig], Generic[T_ModelPipelineConfig]
):
    __abstract__ = True
    __config__: type[T_ModelPipelineConfig]

    def __init__(self, config: T_ModelPipelineConfig, labels: DatasetLabels) -> None:
        from atria_models.core.model_builders._base import ModelBuilder
        from atria_models.core.model_pipelines._ops import ModelPipelineOps
        from atria_models.core.model_pipelines._state_handler import StateDictHandler

        super().__init__(config=config)

        self._labels = labels
        self._state_dict_handler = StateDictHandler(self)
        self._ops = ModelPipelineOps(self)
        self._model_builder = ModelBuilder.from_type(
            builder_type=self.config.model.builder_type,
            bn_to_gn=self.config.model.bn_to_gn,
            frozen_layers=self.config.model.frozen_layers,
            pretrained_checkpoint=self.config.model.pretrained_checkpoint,
        )

        assert self.config.model.model_name_or_path is not None, (
            "ModelPipelineConfig.model.model_name_or_path must be specified to build the model."
            f" Current config: {self.config}"
        )
        self._model = self._model_builder.build(
            model_name_or_path=self.config.model.model_name_or_path,
            **(self.config.model.model_kwargs or {}),
            **self._model_build_kwargs(),
        )
        self._model_args_list = []
        for param in inspect.signature(self._model.forward).parameters.values():
            if param.name != "self":
                self._model_args_list.append(param.name)

    @property
    def ops(self) -> ModelPipelineOps:
        return self._ops

    @property
    def ema_modules(self) -> torch.nn.Module:
        return self._model

    @property
    def trainable_parameters(self) -> dict[str, Iterator[torch.nn.Parameter]]:
        return self.ops.get_trainable_parameters()

    def state_dict(self) -> dict:
        return self._state_dict_handler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._state_dict_handler.load_state_dict(state_dict)

    def _model_build_kwargs(self) -> dict[str, object]:
        return {}

    def build_metrics(
        self,
        stage: Literal["train", "validation", "test"],
        device: torch.device | str = "cpu",
    ) -> dict[str, Metric]:
        return {}

    def training_step(
        self, batch: TensorDataModel, training_engine: Engine | None = None, **kwargs
    ) -> ModelOutput:
        raise NotImplementedError(
            "Training step is not implemented for this model pipeline."
        )

    def evaluation_step(
        self,
        batch: TensorDataModel,
        evaluation_engine: Engine | None = None,
        training_engine: Engine | None = None,
        stage: Literal["validation", "test"] = "test",
        **kwargs,
    ) -> ModelOutput:
        raise NotImplementedError(
            "Evaluation step is not implemented for this model pipeline."
        )

    def predict_step(
        self, batch: TensorDataModel, prediction_engine: Engine | None = None, **kwargs
    ) -> ModelOutput:
        raise NotImplementedError(
            "Prediction step is not implemented for this model pipeline."
        )

    def visualization_step(
        self,
        batch: TensorDataModel,
        visualization_engine: Engine | None = None,
        training_engine: Engine | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            "Visualization step is not implemented for this model pipeline."
        )
