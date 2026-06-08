
import inspect
from typing import Any

import torch
from atria_logger import get_logger
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_models.core.models.transformers._outputs import (
    SequenceClassificationHeadOutput,
    TransformersEncoderModelOutput,
)
from atria_models.core.types.model_outputs import ClassificationModelOutput

logger  = get_logger(__name__)

class ImageModelExplanationForwardWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self._model = model
        self._model_signature = inspect.signature(model.forward)
        assert isinstance(self._model, torch.nn.Module), (
            f"{self.__class__.__name__} only supports torch.nn.Module"
        )

    def _sanitize_inputs(self, *args) -> dict[str, Any]:
        # in sequence forward warppres we do args[-1][0] since we sent a list of args mapping corresponding to each sample in batch
        # but its not necessary
        args_mapping = args[-1]
        assert isinstance(args_mapping, list), (
            f"Expected args_mapping to be a list of keys, got {type(args_mapping)}"
        )
        args = args[:-1]  # all but last are the actual model args
        logger.debug(
            f"WrappedModel.forward called with {len(args)} args and args_mapping: {args_mapping}"
        )

        if len(args) != len(args_mapping):
            raise ValueError(
                f"Expected {len(args_mapping)} inputs, but got {len(args)} inputs."
            )

        model_kwargs = {args_mapping[i]: args[i] for i in range(len(args_mapping))}

        # if 'x' exists in self._model_signature.parameters we map image to x
        if 'x' in self._model_signature.parameters:
            model_kwargs['x'] = model_kwargs.pop('image')
        else:
            # filter unsupported args (important for heterogeneous models)
            model_kwargs = {
                k: v
                for k, v in model_kwargs.items()
                if k in self._model_signature.parameters
            }
        return model_kwargs

    def forward(self, *args) -> torch.Tensor:
        model_kwargs = self._sanitize_inputs(*args)

        for key, value in model_kwargs.items():
            logger.debug(
                f"Model input - {key}: {value.shape if hasattr(value, 'shape') else value}"
            )
        logits = self._model(**model_kwargs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs