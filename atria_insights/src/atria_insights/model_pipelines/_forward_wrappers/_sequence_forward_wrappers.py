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

logger = get_logger(__name__)


class ExplainableSequenceModelForwardWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model
        self._model_signature = inspect.signature(model.forward)
        assert isinstance(self._model, TransformersEncoderModel), (
            f"{self.__class__.__name__} only supports TransformersEncoderModel"
        )

    def _sanitize_inputs(self, *args) -> dict[str, Any]:
        args_mapping = args[-1]  # the last arg is the args mapping
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

        # we remap embedding args to id_or_embedding args
        # inside the model, we need to remap ids -> embeddings
        for key in ["token_ids", "position_ids", "layout_ids", "token_type_ids"]:
            if key in model_kwargs:
                model_kwargs[key.replace("_ids", "_ids_or_embeddings")] = (
                    model_kwargs.pop(key)
                )

        # filter unsupported args (important for heterogeneous models)
        model_kwargs = {
            k: v
            for k, v in model_kwargs.items()
            if k in self._model_signature.parameters
        }

        return model_kwargs

    def forward(self, *args) -> torch.Tensor:
        model_kwargs = self._sanitize_inputs(*args)
        outputs = self._model(**model_kwargs, is_embedding=True)
        assert isinstance(outputs, TransformersEncoderModelOutput)
        assert isinstance(outputs.head_output, SequenceClassificationHeadOutput)
        assert outputs.head_output.logits is not None
        return torch.nn.functional.softmax(outputs.head_output.logits, dim=-1)
