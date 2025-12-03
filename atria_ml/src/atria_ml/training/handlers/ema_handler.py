"""
Atria EMA Handler Module

This module defines the `AtriaEMAHandler` class, which extends the functionality of the
`ignite.handlers.EMAHandler` to include additional features for managing Exponential Moving
Average (EMA) models in the Atria framework.

Classes:
    - AtriaEMAHandler: Custom handler for managing EMA models.

Dependencies:
    - torch: For PyTorch operations.
    - ignite: For engine and handler management.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

import torch
from atria_core.logger import get_logger
from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList
from ignite.handlers import EMAHandler
from ignite.handlers.state_param_scheduler import LambdaStateScheduler

logger = get_logger(__name__)


class AtriaEMAHandler(EMAHandler):
    """
    Custom handler for managing Exponential Moving Average (EMA) models.

    This handler extends the `ignite.handlers.EMAHandler` to provide additional functionality
    for attaching to engines and swapping model parameters.
    """

    def attach(
        self,
        engine: Engine,
        name: str = "ema_momentum",
        warn_if_exists: bool = True,
        event: str
        | Events
        | CallableEventWithFilter
        | EventsList = Events.ITERATION_COMPLETED,
    ) -> None:
        """
        Attach the handler to an engine.

        After the handler is attached, the `Engine.state` will include a new attribute with the
        specified name, allowing retrieval of the current EMA momentum.

        Args:
            engine (Engine): The engine to which the handler will be attached.
            name (str): Attribute name for retrieving EMA momentum from `Engine.state`. It should
                be unique since an engine can have multiple EMA handlers.
            warn_if_exists (bool): If True, a warning is issued if the momentum with the specified
                name already exists.
            event (Union[str, Events, CallableEventWithFilter, EventsList]): The event during which
                the EMA momentum and EMA model are updated.

        Note:
            - If the attribute with the specified name already exists, it may be due to the engine
              loading its state dict or another handler creating the same attribute. In such cases,
              set `warn_if_exists` to False to suppress the warning.
        """
        if hasattr(engine.state, name):
            if warn_if_exists:
                logger.warning(
                    f"Attribute '{name}' already exists in Engine.state. It might because 1. the engine has loaded its "
                    f"state dict or 2. {name} is already created by other handlers. Turn off this warning by setting"
                    f"warn_if_exists to False.",
                    category=UserWarning,
                )
        else:
            setattr(engine.state, name, self.momentum)

        if self._momentum_lambda_obj is not None:
            self.momentum_scheduler = LambdaStateScheduler(
                self._momentum_lambda_obj, param_name=name
            )

            # first update the momentum and then update the EMA model
            self.momentum_scheduler.attach(engine, event)
        engine.add_event_handler(event, self._update_ema_model, name)

    @torch.no_grad()
    def swap_params(self) -> None:
        """
        Swap the parameters between the EMA model and the original model.

        This method exchanges the parameters of the EMA model with those of the original model,
        allowing evaluation or other operations to be performed using the EMA model's parameters.
        """
        for ema_v, model_v in zip(
            self.ema_model.state_dict().values(),
            self.model.state_dict().values(),
            strict=True,
        ):
            tmp = model_v.data.clone()
            model_v.data.copy_(ema_v.data)
            ema_v.data.copy_(tmp)
