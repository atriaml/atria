from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Generic

from atria_logger import get_logger
from atria_registry._module_base import RegisterableModule
from atria_types._common import TrainingStage
from atria_types._datasets import DatasetLabels

from atria_models.core.model_pipelines._common import T_ModelPipelineConfig
from atria_models.core.types.model_outputs import ModelOutput

if TYPE_CHECKING:
    import torch
    from atria_types import BaseDataInstance
    from ignite.engine import Engine

    from atria_models.core.model_pipelines._ops import ModelPipelineOps


logger = get_logger(__name__)


class ModelPipeline(
    RegisterableModule[T_ModelPipelineConfig], Generic[T_ModelPipelineConfig]
):
    __abstract__ = True
    __config__: type[T_ModelPipelineConfig]

    def __init__(
        self,
        labels: DatasetLabels,
        config: T_ModelPipelineConfig | None = None,
        **config_overrides: Any,
    ) -> None:
        from atria_models.core.model_builders._base import ModelBuilder
        from atria_models.core.model_pipelines._ops import ModelPipelineOps
        from atria_models.core.model_pipelines._state_handler import StateDictHandler

        super().__init__(config=config, **config_overrides)

        self._labels = labels
        self._state_dict_handler = StateDictHandler(self)
        self._ops = ModelPipelineOps(self)
        self._model_builder = ModelBuilder.from_type(
            builder_type=self.config.model_config.builder_type,
            bn_to_gn=self.config.model_config.bn_to_gn,
            frozen_layers=self.config.model_config.frozen_layers,
            pretrained_checkpoint=self.config.model_config.pretrained_checkpoint,
        )
        self._model = self._model_builder.build(
            model_name_or_path=self.config.model_config.model_name_or_path,
            **(self.config.model_config.model_kwargs or {}),
            **self._model_build_kwargs(),
        )
        self._model_args_list = []
        for param in inspect.signature(self._model.forward).parameters.values():
            if param.name != "self":
                self._model_args_list.append(param.name)

    @property
    def pipeline_name(self) -> str:
        assert self.config.name is not None, "Pipeline config must have a name."
        return self.config.name

    @property
    def ops(self) -> ModelPipelineOps:
        return self._ops

    @property
    def ema_modules(self) -> torch.nn.Module:
        return self._model

    @property
    def trainable_parameters(self) -> dict[str, list[torch.nn.Parameter]]:
        return self.ops.get_trainable_parameters()

    def _model_build_kwargs(self) -> dict[str, object]:
        return {}

    def training_step(
        self, batch: BaseDataInstance, training_engine: Engine | None = None, **kwargs
    ) -> ModelOutput:
        raise NotImplementedError(
            "Training step is not implemented for this model pipeline."
        )

    def evaluation_step(
        self,
        batch: BaseDataInstance,
        evaluation_engine: Engine | None = None,
        training_engine: Engine | None = None,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> ModelOutput:
        raise NotImplementedError(
            "Evaluation step is not implemented for this model pipeline."
        )

    def predict_step(
        self, batch: BaseDataInstance, evaluation_engine: Engine | None = None, **kwargs
    ) -> ModelOutput:
        raise NotImplementedError(
            "Prediction step is not implemented for this model pipeline."
        )

    def visualization_step(
        self,
        batch: BaseDataInstance,
        evaluation_engine: Engine | None = None,
        training_engine: Engine | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            "Visualization step is not implemented for this model pipeline."
        )

    def __repr__(self):
        return f"{self.pipeline_name}:\n{self.ops.summarize()}"

    def __str__(self):
        return self.__repr__()
