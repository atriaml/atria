from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from atria_models.core.types.model_outputs import ModelOutput
from atria_transforms.core._data_types._base import TensorDataModel

from atria_ml.training.engine_steps._base import EngineStep

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine
    from torch.optim import Optimizer

    from atria_ml.training._configs import GradientConfig

logger = get_logger(__name__)


class TrainingStep(EngineStep):
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        device: str | torch.device,
        optimizers: dict[str, Optimizer],
        gradient_config: GradientConfig,
        grad_scaler: torch.cuda.amp.grad_scaler.GradScaler | None = None,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        super().__init__(
            model_pipeline=model_pipeline,
            device=device,
            with_amp=with_amp,
            test_run=test_run,
        )

        from torch.cuda.amp.grad_scaler import GradScaler

        self._optimizers = optimizers
        self._gradient_config = gradient_config
        self._grad_scaler = (
            GradScaler(enabled=True) if grad_scaler is None else grad_scaler
        )

    @property
    def stage(self) -> str:
        return "train"

    def _validate_gradient_config(self) -> None:
        if self._gradient_config.gradient_accumulation_steps <= 0:
            raise ValueError(
                "Gradient_accumulation_steps must be strictly positive. "
                "No gradient accumulation if the value set to one (default)."
            )

    def _reset_optimizers(self, engine: Engine) -> None:
        if (
            engine.state.iteration - 1
        ) % self._gradient_config.gradient_accumulation_steps == 0:
            for opt in self._optimizers.values():
                opt.zero_grad()

    def _call_forward(self, engine: Engine, batch: TensorDataModel) -> ModelOutput:
        from torch.amp.autocast_mode import autocast

        with autocast(device_type=self._device.type, enabled=self._with_amp):
            # forward pass
            model_output = self._model_pipeline.training_step(
                training_engine=engine, batch=batch, test_run=self._test_run
            )

            # make sure we get a dict from the model
            assert isinstance(model_output, ModelOutput), (
                f"Model must return an instance of ModelOutput. Current type: {type(model_output)}"
            )
            assert model_output.loss is not None, (
                "Model output 'loss' must not be None during the training step. "
            )

            return replace(
                model_output,
                loss=model_output.loss
                / self._gradient_config.gradient_accumulation_steps,
            )

    def _update_optimizers_with_grad_scaler(
        self, engine: Engine, loss: torch.Tensor, optimizer_key: str | None = None
    ) -> None:
        """
        Updates the optimizers using the GradScaler for AMP.

        Args:
            engine (Engine): The engine executing this step.
            loss (torch.Tensor): The loss tensor.
            optimizer_key (Optional[str]): The key of the optimizer to update. Defaults to None.
        """
        self._grad_scaler.scale(loss).backward()  # type: ignore

        # perform optimizer update for correct gradient accumulation step
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            # perform gradient clipping if needed
            if self._gradient_config.enable_grad_clipping:
                # Unscales the gradients of optimizer's assigned params in-place
                for key, opt in self._optimizers.items():
                    if optimizer_key is None or key == optimizer_key:
                        self._grad_scaler.unscale_(opt)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                trainable_parameters_concat = []
                for params in self._model_pipeline.trainable_parameters.values():
                    trainable_parameters_concat.extend(list(params))
                torch.nn.utils.clip_grad_norm_(  # type: ignore
                    trainable_parameters_concat, self._gradient_config.max_grad_norm
                )

            for key, opt in self._optimizers.items():
                if optimizer_key is None or key == optimizer_key:
                    self._grad_scaler.step(opt)

            # scaler update should be called only once. See https://pytorch.org/docs/stable/amp.html
            self._grad_scaler.update()

    def _update_optimizers_standard(
        self, engine: Engine, loss: torch.Tensor, optimizer_key: str | None = None
    ) -> None:
        """
        Updates the optimizers without using the GradScaler.

        Args:
            engine (Engine): The engine executing this step.
            loss (torch.Tensor): The loss tensor.
            optimizer_key (Optional[str]): The key of the optimizer to update. Defaults to None.
        """
        # backward pass
        loss.backward()

        # perform optimizer update for correct gradient accumulation step
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            # perform gradient clipping if needed
            if self._gradient_config.enable_grad_clipping:
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                trainable_parameters_concat = []
                for params in self._model_pipeline.trainable_parameters.values():
                    trainable_parameters_concat.extend(list(params))
                torch.nn.utils.clip_grad_norm_(  # type: ignore
                    trainable_parameters_concat, self._gradient_config.max_grad_norm
                )

            for key, opt in self._optimizers.items():
                if optimizer_key is None or key == optimizer_key:
                    opt.step()

    def _update_optimizers(
        self, engine: Engine, loss: torch.Tensor, optimizer_key: str | None = None
    ) -> None:
        """
        Updates the optimizers based on whether AMP is enabled.

        Args:
            engine (Engine): The engine executing this step.
            loss (torch.Tensor): The loss tensor.
            optimizer_key (Optional[str]): The key of the optimizer to update. Defaults to None.
        """
        if self._grad_scaler:
            self._update_optimizers_with_grad_scaler(
                engine=engine, loss=loss, optimizer_key=optimizer_key
            )
        else:
            self._update_optimizers_standard(
                engine=engine, loss=loss, optimizer_key=optimizer_key
            )

    def __call__(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        from atria_ml.training.engines._events import OptimizerEvents

        self._validate_gradient_config()
        self._reset_optimizers(engine=engine)
        self._model_pipeline.ops.train()
        batch = batch.ops.to(self._device)
        model_output = self._call_forward(engine=engine, batch=batch)
        self._update_optimizers(engine=engine, loss=model_output.loss)  # type: ignore
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            engine.fire_event(OptimizerEvents.optimizer_step)
            engine.state.optimizer_step += 1  # type: ignore

        return model_output
