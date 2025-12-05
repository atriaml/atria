import torch
from atria_logger import get_logger
from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList
from ignite.handlers import EMAHandler as IgniteEMAHandler
from ignite.handlers.state_param_scheduler import LambdaStateScheduler

logger = get_logger(__name__)


class EMAHandler(IgniteEMAHandler):
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
        for ema_v, model_v in zip(
            self.ema_model.state_dict().values(),
            self.model.state_dict().values(),
            strict=True,
        ):
            tmp = model_v.data.clone()
            model_v.data.copy_(ema_v.data)
            ema_v.data.copy_(tmp)
