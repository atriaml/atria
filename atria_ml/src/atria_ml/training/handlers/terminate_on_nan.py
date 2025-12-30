import numbers

import torch
from atria_transforms.core._data_types._base import TensorDataModel
from ignite.engine import Engine
from ignite.handlers import TerminateOnNan as IgniteTerminateOnNan
from ignite.utils import apply_to_type
from pydantic import BaseModel


class TerminateOnNan(IgniteTerminateOnNan):
    def __call__(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)

        def raise_error(x: float | torch.Tensor) -> None:
            if x is None or isinstance(x, (TensorDataModel)):
                return

            if isinstance(x, numbers.Number):
                x = torch.tensor(x)

            if isinstance(x, torch.Tensor) and not bool(torch.isfinite(x).all()):
                raise RuntimeError("Infinite or NaN tensor found.")

        try:
            output = output.model_dump() if isinstance(output, BaseModel) else output
            apply_to_type(output, (numbers.Number, torch.Tensor), raise_error)
        except RuntimeError:
            self.logger.warning(
                f"{self.__class__.__name__}: Output '{output}' contains NaN or Inf. Stop training"
            )
            engine.terminate()
