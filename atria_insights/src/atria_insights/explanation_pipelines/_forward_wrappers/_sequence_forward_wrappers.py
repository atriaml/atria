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


class SequenceModelExplanationForwardWrapper(torch.nn.Module):
    def __init__(
        self,
        model: TransformersEncoderModel,
        is_embedding: bool = True,
        return_attns: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self._model_signature = inspect.signature(model.forward)
        self._is_embedding = is_embedding
        self._return_attns = return_attns
        assert isinstance(self._model, TransformersEncoderModel), (
            f"{self.__class__.__name__} only supports TransformersEncoderModel"
        )
        self._model.config.unsafe_update(output_attentions=return_attns)

    @property
    def return_attns(self) -> bool:
        return self._return_attns

    @return_attns.setter
    def return_attns(self, value: bool) -> None:
        self._return_attns = value
        self._model.config.unsafe_update(output_attentions=value)

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
        for key in ["token_ids", "position_ids", "layout_ids", "token_type_ids"]:
            if key in model_kwargs:
                model_kwargs[key.replace("_ids", "_ids_or_embeddings")] = (
                    model_kwargs.pop(key)
                    if key not in self._model_signature.parameters
                    else model_kwargs[key]
                )

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
        outputs = self._model(**model_kwargs, is_embedding=self._is_embedding)
        assert isinstance(outputs, TransformersEncoderModelOutput)
        assert isinstance(outputs.head_output, SequenceClassificationHeadOutput)
        assert outputs.head_output.logits is not None
        probs = torch.nn.functional.softmax(outputs.head_output.logits, dim=-1)
        if self._return_attns:
            assert outputs.attentions is not None, (
                "Model output contains no attentions. Make sure the model is configured to output attentions."
            )
            return outputs.attentions
        return probs


class TokenClassificationModelExplanationForwardWrapper(
    SequenceModelExplanationForwardWrapper
):
    def forward(self, *args) -> torch.Tensor:
        model_kwargs = self._sanitize_inputs(*args)
        for key, value in model_kwargs.items():
            logger.debug(
                f"Model input - {key}: {value.shape if hasattr(value, 'shape') else value}"
            )

        outputs = self._model(**model_kwargs, is_embedding=self._is_embedding)
        assert isinstance(outputs, TransformersEncoderModelOutput)
        assert isinstance(outputs.head_output, TokenClassificationHeadOutput)
        assert outputs.head_output.logits is not None
        probs = torch.nn.functional.softmax(outputs.head_output.logits, dim=-1)
        probs = torch.gather(probs, 2, probs.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)

        if self._return_attns:
            assert outputs.attentions is not None, (
                "Model output contains no attentions. Make sure the model is configured to output attentions."
            )
            return outputs.attentions

        return probs


class QuestionAnsweringModelExplanationForwardWrapper(
    SequenceModelExplanationForwardWrapper
):
    def forward(self, *args) -> torch.Tensor:
        model_kwargs = self._sanitize_inputs(*args)
        for key, value in model_kwargs.items():
            logger.debug(
                f"Model input - {key}: {value.shape if hasattr(value, 'shape') else value}"
            )

        outputs = self._model(**model_kwargs, is_embedding=self._is_embedding)
        assert isinstance(outputs, TransformersEncoderModelOutput)
        assert isinstance(outputs.head_output, QuestionAnsweringHeadOutput)
        assert outputs.head_output.start_logits is not None
        assert outputs.head_output.end_logits is not None
        start_probs = torch.nn.functional.softmax(
            outputs.head_output.start_logits, dim=-1
        )
        # start_pred_prob = start_probs[
        #     torch.arange(start_probs.size(0)), start_probs.argmax(dim=-1)
        # ]
        end_probs = torch.nn.functional.softmax(outputs.head_output.end_logits, dim=-1)
        # end_pred_prob = end_probs[
        #     torch.arange(end_probs.size(0)), end_probs.argmax(dim=-1)
        # ]

        # probs = torch.cat(
        #     [start_pred_prob.unsqueeze(-1), end_pred_prob.unsqueeze(-1)], dim=-1
        # )
        if self._return_attns:
            assert outputs.attentions is not None, (
                "Model output contains no attentions. Make sure the model is configured to output attentions."
            )
            return outputs.attentions
        return torch.stack([start_probs, end_probs], dim=1)
