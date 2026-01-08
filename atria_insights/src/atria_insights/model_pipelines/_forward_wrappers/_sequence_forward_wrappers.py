import inspect
from typing import Any

import torch
from atria_logger import get_logger
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_models.core.models.transformers._outputs import (
    QuestionAnsweringHeadOutput,
    SequenceClassificationHeadOutput,
    TokenClassificationHeadOutput,
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
        args_mapping = args[-1][0]  # the last arg is the args mapping
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
        for key in [
            "token_embeddings",
            "position_embeddings",
            "layout_embeddings",
            "token_type_embeddings",
        ]:
            if key in model_kwargs:
                model_kwargs[key.replace("_embeddings", "_ids_or_embeddings")] = (
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
        for key, value in model_kwargs.items():
            logger.debug(
                f"Model input - {key}: {value.shape if hasattr(value, 'shape') else value}"
            )
        outputs = self._model(**model_kwargs, is_embedding=True)
        assert isinstance(outputs, TransformersEncoderModelOutput)
        assert isinstance(outputs.head_output, SequenceClassificationHeadOutput)
        assert outputs.head_output.logits is not None
        return torch.nn.functional.softmax(outputs.head_output.logits, dim=-1)


class ExplainableTokenClassificationModelForwardWrapper(
    ExplainableSequenceModelForwardWrapper
):
    def forward(self, *args) -> torch.Tensor:
        model_kwargs = self._sanitize_inputs(*args)
        for key, value in model_kwargs.items():
            logger.debug(
                f"Model input - {key}: {value.shape if hasattr(value, 'shape') else value}"
            )
        outputs = self._model(**model_kwargs, is_embedding=True)
        assert isinstance(outputs, TransformersEncoderModelOutput)
        assert isinstance(outputs.head_output, TokenClassificationHeadOutput)
        assert outputs.head_output.logits is not None
        probs = torch.nn.functional.softmax(outputs.head_output.logits)
        probs = torch.gather(probs, 2, probs.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)
        return probs


class ExplainableQuestionAnsweringModelForwardWrapper(
    ExplainableSequenceModelForwardWrapper
):
    def forward(self, *args) -> torch.Tensor:
        model_kwargs = self._sanitize_inputs(*args)
        for key, value in model_kwargs.items():
            logger.debug(
                f"Model input - {key}: {value.shape if hasattr(value, 'shape') else value}"
            )
        outputs = self._model(**model_kwargs, is_embedding=True)
        assert isinstance(outputs, TransformersEncoderModelOutput)
        assert isinstance(outputs.head_output, QuestionAnsweringHeadOutput)
        assert outputs.head_output.start_logits is not None
        assert outputs.head_output.end_logits is not None
        start_probs = torch.nn.functional.softmax(
            outputs.head_output.start_logits, dim=-1
        )
        start_pred_prob = start_probs[
            torch.arange(start_probs.size(0)), start_probs.argmax(dim=-1)
        ]
        end_probs = torch.nn.functional.softmax(outputs.head_output.end_logits, dim=-1)
        end_pred_prob = end_probs[
            torch.arange(end_probs.size(0)), end_probs.argmax(dim=-1)
        ]
        return torch.cat(
            [start_pred_prob.unsqueeze(-1), end_pred_prob.unsqueeze(-1)], dim=-1
        )
