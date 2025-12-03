"""
This module defines the training steps for the training engine.

The `TrainingStep` class provides the default implementation for a training step, including support for gradient accumulation, AMP (automatic mixed precision), and optimizer updates. The `GANTrainingStep` class extends this functionality to support GAN-specific training logic, including separate optimizers for generator and discriminator.

Classes:
    TrainingStep: Default implementation of a training step.
    GANTrainingStep: Implementation of a training step for GANs.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

from atria_core.logger.logger import get_logger

from atria_ml.training.engines.engine_steps.base import BaseEngineStep
from atria_ml.training.utilities.constants import GANStage, TrainingStage

if TYPE_CHECKING:
    import torch
    from atria_core.types import BaseDataInstance
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
    from ignite.engine import Engine
    from torch.optim import Optimizer

    from atria_ml.training.configs.gradient_config import GradientConfig

logger = get_logger(__name__)


class TrainingStep(BaseEngineStep):
    """
    Default implementation of a training step.

    This class handles gradient accumulation, AMP (automatic mixed precision), and optimizer updates during the training process.

    Attributes:
        _model_pipeline (AtriaModelPipeline): The model pipeline to use.
        _device (Union[str, torch.device]): The device to use.
        _optimizers (Dict[str, Optimizer]): The optimizers to use.
        _gradient_config (GradientConfig): The gradient configuration.
        _grad_scaler (Optional[torch.cuda.amp.GradScaler]): GradScaler for AMP. Defaults to None.
        _non_blocking_tensor_conv (bool): Whether tensor conversion should be non-blocking. Defaults to False.
        _with_amp (bool): Whether AMP is enabled. Defaults to False.
        _test_run (bool): Whether this is a test run. Defaults to False.
    """

    def __init__(
        self,
        model_pipeline: "AtriaModelPipeline",
        device: Union[str, "torch.device"],
        optimizers: dict[str, "Optimizer"],
        gradient_config: "GradientConfig",
        grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        """
        Initializes the TrainingStep instance.

        Args:
            model_pipeline (AtriaModelPipeline): The model pipeline to use.
            device (Union[str, torch.device]): The device to use.
            optimizers (Dict[str, Optimizer]): The optimizers to use.
            gradient_config (GradientConfig): The gradient configuration.
            grad_scaler (Optional[torch.cuda.amp.GradScaler]): GradScaler for AMP. Defaults to None.
            with_amp (bool): Whether AMP is enabled. Defaults to False.
            test_run (bool): Whether this is a test run. Defaults to False.
        """
        super().__init__()

        import torch

        self._model_pipeline = model_pipeline
        self._device = torch.device(device)
        self._optimizers = optimizers
        self._gradient_config = gradient_config
        self._grad_scaler = grad_scaler
        self._with_amp = with_amp
        self._test_run = test_run
        self._setup_amp()

    @property
    def stage(self) -> TrainingStage:
        """
        Returns the training stage for this step.

        Returns:
            TrainingStage: The training stage.
        """
        return TrainingStage.train

    @property
    def gradient_config(self) -> "GradientConfig":
        """
        Returns the gradient configuration for this step.

        Returns:
            GradientConfig: The gradient configuration.
        """
        return self._gradient_config

    def _setup_amp(self) -> None:
        """
        Sets up AMP (automatic mixed precision) if enabled.

        Raises:
            ImportError: If AMP is enabled but the required PyTorch version is not installed.
        """
        if self._with_amp:
            try:
                pass
            except ImportError:
                raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

            from torch.cuda.amp.grad_scaler import GradScaler

            self._grad_scaler = GradScaler(enabled=True)

    def _validate_gradient_config(self) -> None:
        """
        Validates the gradient configuration.

        Raises:
            ValueError: If gradient accumulation steps are not strictly positive.
        """
        if self._gradient_config.gradient_accumulation_steps <= 0:
            raise ValueError(
                "Gradient_accumulation_steps must be strictly positive. "
                "No gradient accumulation if the value set to one (default)."
            )

    def _reset_optimizers(self, engine: "Engine") -> None:
        """
        Resets the optimizers for gradient accumulation.

        Args:
            engine (Engine): The engine executing this step.
        """
        # perform optimizers zero_grad() operation with gradient accumulation
        if (
            engine.state.iteration - 1
        ) % self._gradient_config.gradient_accumulation_steps == 0:
            for opt in self._optimizers.values():
                opt.zero_grad()

    def _call_forward(self, engine: "Engine", batch: "BaseDataInstance") -> None:
        """
        Performs the forward pass for the training step.

        Args:
            engine (Engine): The engine executing this step.
            batch (BaseDataInstance): The batch of data to process.

        Returns:
            Tuple[torch.Tensor, ModelOutput]: The loss and model output.
        """
        from atria_core.types import ModelOutput
        from torch.amp import autocast

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

            # get the loss
            loss = model_output.loss

            # accumulate loss if required
            if self._gradient_config.gradient_accumulation_steps > 1:
                loss = loss / self._gradient_config.gradient_accumulation_steps

            return loss, model_output

    def _update_optimizers_with_grad_scaler(
        self, engine: "Engine", loss: "torch.Tensor", optimizer_key: str | None = None
    ) -> None:
        """
        Updates the optimizers using the GradScaler for AMP.

        Args:
            engine (Engine): The engine executing this step.
            loss (torch.Tensor): The loss tensor.
            optimizer_key (Optional[str]): The key of the optimizer to update. Defaults to None.
        """
        self._grad_scaler.scale(loss).backward()

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
                torch.nn.utils.clip_grad_norm_(
                    self._model_pipeline.torch_model.parameters(),
                    self._gradient_config.max_grad_norm,
                )

            for key, opt in self._optimizers.items():
                if optimizer_key is None or key == optimizer_key:
                    self._grad_scaler.step(opt)

            # scaler update should be called only once. See https://pytorch.org/docs/stable/amp.html
            self._grad_scaler.update()

    def _update_optimizers_standard(
        self, engine: "Engine", loss: "torch.Tensor", optimizer_key: str | None = None
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
                torch.nn.utils.clip_grad_norm_(
                    self._model_pipeline.torch_model.parameters(),
                    self._gradient_config.max_grad_norm,
                )

            for key, opt in self._optimizers.items():
                if optimizer_key is None or key == optimizer_key:
                    opt.step()

    def _update_optimizers(
        self, engine: "Engine", loss: "torch.Tensor", optimizer_key: str | None = None
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
        self, engine: "Engine", batch: "BaseDataInstance"
    ) -> Any | tuple["torch.Tensor"]:
        """
        Executes the training step.

        Args:
            engine (Engine): The engine executing this step.
            batch (BaseDataInstance): The batch of data to process.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the training step.
        """
        from atria_ml.training.engines.events import OptimizerEvents

        self._validate_gradient_config()
        self._reset_optimizers(engine=engine)
        self._model_pipeline.train()
        if hasattr(batch, "to_device"):
            batch = batch.to_device(self._device)
        loss, model_output = self._call_forward(engine=engine, batch=batch)
        self._update_optimizers(engine=engine, loss=loss)
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            # call update
            engine.fire_event(OptimizerEvents.optimizer_step)
            engine.state.optimizer_step += 1

        return model_output


class GANTrainingStep(TrainingStep):
    """
    Implementation of a training step for GANs.

    This class extends the `TrainingStep` class to support GAN-specific training logic, including separate optimizers for generator and discriminator.
    """

    def __init__(
        self,
        model_pipeline: "AtriaModelPipeline",
        device: Union[str, "torch.device"],
        optimizers: dict[str, "Optimizer"],
        gradient_config: "GradientConfig",
        grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        """
        Initializes the GANTrainingStep instance.

        Args:
            model_pipeline (AtriaModelPipeline): The model pipeline to use.
            device (Union[str, torch.device]): The device to use.
            optimizers (Dict[str, Optimizer]): The optimizers to use.
            gradient_config (GradientConfig): The gradient configuration.
            grad_scaler (Optional[torch.cuda.amp.GradScaler]): GradScaler for AMP. Defaults to None.
            with_amp (bool): Whether AMP is enabled. Defaults to False.
            test_run (bool): Whether this is a test run. Defaults to False.
        """
        super().__init__(
            model_pipeline=model_pipeline,
            device=device,
            optimizers=optimizers,
            gradient_config=gradient_config,
            grad_scaler=grad_scaler,
            with_amp=with_amp,
            test_run=test_run,
        )

    def _validate_optimizers(self) -> None:
        """
        Validates the optimizers for GAN training.

        Raises:
            AssertionError: If required optimizers are not defined.
        """
        # make sure that optimizers are set up for generator and discriminator
        assert "generator" in self._optimizers.keys(), (
            f"You must define and optimizer with key 'generator' to use the {self.__class__.__name__}. "
        )
        assert "discriminator" in self._optimizers.keys(), (
            f"You must define and optimizer with key 'discriminator' to use the {self.__class__.__name__}. "
        )

    def _toggle_optimizer(self, opt: "torch.optim.Optimizer") -> None:
        """
        Toggles the optimizer by enabling or disabling its parameters.

        Args:
            opt (torch.optim.Optimizer): The optimizer to toggle.
        """
        # disable all parameters
        for param in self._model_pipeline.torch_model.parameters():
            param.requires_grad = False

        # enable the parameters of the optimizer
        for group in opt.param_groups:
            for param in group["params"]:
                param.requires_grad = True

        # # print the parameters that are enabled
        # for name, param in self._model_pipeline.torch_model.named_parameters():
        #     if param.requires_grad:
        #         logger.debug(f"Enabled parameter: {name}")

    def _call_forward(
        self, engine: "Engine", batch: dict[str, "torch.Tensor"], gan_stage: "GANStage"
    ) -> None:
        """
        Performs the forward pass for the specified GAN stage.

        Args:
            engine (Engine): The engine executing this step.
            batch (Dict[str, torch.Tensor]): The batch of data to process.
            gan_stage (GANStage): The GAN stage (generator or discriminator).

        Returns:
            Tuple[torch.Tensor, ModelOutput]: The loss and model output.
        """
        from atria_core.types import ModelOutput
        from torch.amp import autocast

        with autocast(device_type=self._device.type, enabled=self._with_amp):
            # forward pass
            model_output = self._model_pipeline.training_step(
                training_engine=engine,
                batch=batch,
                test_run=self._test_run,
                gan_stage=gan_stage,
            )

            # make sure we get a dict from the model
            assert isinstance(model_output, ModelOutput), (
                "Model must return an instance of ModelOutput."
            )
            assert model_output.loss is not None, (
                "Model output loss must not be None during the training step. "
            )

            # get the loss
            loss = model_output.loss

            # accumulate loss if required
            if self._gradient_config.gradient_accumulation_steps > 1:
                loss = loss / self._gradient_config.gradient_accumulation_steps

            return loss, model_output

    def __call__(
        self, engine: "Engine", batch: "BaseDataInstance"
    ) -> Any | tuple["torch.Tensor"]:
        """
        Executes the GAN training step.

        Args:
            engine (Engine): The engine executing this step.
            batch (BaseDataInstance): The batch of data to process.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the GAN training step.
        """
        from atria_ml.training.engines.events import OptimizerEvents
        from atria_ml.training.utilities.constants import GANStage

        self._validate_gradient_config()
        self._reset_optimizers(engine=engine)
        self._model_pipeline.train()
        batch = batch.to_device(self._device)
        step_outputs = {}
        for gan_stage in [GANStage.train_generator, GANStage.train_discriminator]:
            # only train the generator or discriminator depending upon the stage
            optimizer_key = (
                "generator"
                if gan_stage == GANStage.train_generator
                else "discriminator"
            )

            # only compute gradients for this optimizer parameters
            self._toggle_optimizer(self._optimizers[optimizer_key])

            # perform forward pass for the given stage
            loss, model_output = self._call_forward(
                engine=engine, batch=batch, gan_stage=gan_stage
            )
            self._update_optimizers(
                engine=engine, loss=loss, optimizer_key=optimizer_key
            )

            output_key = "gen" if gan_stage == GANStage.train_generator else "disc"
            step_outputs[f"{output_key}_loss"] = model_output["loss"]
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            # call update
            engine.fire_event(OptimizerEvents.optimizer_step)
            engine.state.optimizer_step += 1

        return step_outputs
