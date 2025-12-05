from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_models import ModelPipeline
from atria_transforms.core._data_types._base import TensorDataModel
from atria_types._common import TrainingStage
from torch import device
from torch._C import device

from atria_ml.training.engine_steps._base import EngineStep

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine


logger = get_logger(__name__)


class EvaluationStep(EngineStep):
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        device: str | torch.device,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        super().__init__(
            model_pipeline=model_pipeline,
            device=device,
            with_amp=with_amp,
            test_run=test_run,
        )

    def __call__(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        import torch
        from torch.cuda.amp.autocast_mode import autocast

        self._model_pipeline.ops.eval()
        if self._with_amp:
            self._model_pipeline.ops.half()

        with torch.no_grad():
            with autocast(enabled=self._with_amp):
                if hasattr(batch, "to_device"):
                    batch = batch.ops.to(self._device)
                return self._model_step(engine=engine, batch=batch)

    def _model_step(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        raise NotImplementedError("Subclasses must implement this method")


class ValidationStep(EvaluationStep):
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        device: str | device,
        with_amp: bool = False,
        test_run: bool = False,
        training_engine: Engine | None = None,
    ):
        super().__init__(model_pipeline, device, with_amp, test_run)
        self._training_engine = training_engine

    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.validation

    def _model_step(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        return self._model_pipeline.evaluation_step(
            evaluation_engine=engine,
            training_engine=self._training_engine,
            batch=batch,
            stage=self.stage,
        )


class VisualizationStep(EvaluationStep):
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        device: str | device,
        with_amp: bool = False,
        test_run: bool = False,
        training_engine: Engine | None = None,
    ):
        super().__init__(model_pipeline, device, with_amp, test_run)
        self._training_engine = training_engine

    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.visualization

    def _model_step(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        return self._model_pipeline.visualization_step(
            visualization_engine=engine,
            training_engine=self._training_engine,
            batch=batch,
            stage=self.stage,
        )


class TestStep(EvaluationStep):
    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.test

    def _model_step(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        return self._model_pipeline.evaluation_step(
            evaluation_engine=engine, batch=batch, stage=self.stage
        )


class PredictStep(EvaluationStep):
    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.predict

    def _model_step(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        return self._model_pipeline.predict_step(prediction_engine=engine, batch=batch)
