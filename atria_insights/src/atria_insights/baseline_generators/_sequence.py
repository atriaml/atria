from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

from atria_logger import get_logger
from atria_registry._module_base import ModuleConfig

from atria_insights.baseline_generators._base import BaselineGenerator

if TYPE_CHECKING:
    import torch
    from atria_models.core.models.transformers._models._encoder_model import (
        TransformersEncoderModel,
    )

logger = get_logger(__name__)


class SequenceBaselineGeneratorConfig(ModuleConfig):
    module_path: str | None = (
        "atria_insights.baseline_generators._sequence.SequenceBaselineGenerator"
    )
    type: Literal["sequence"] = "sequence"
    text: Literal["zero", "mask_token_id", "pad_token_id"] = "zero"
    token_type: Literal["zero", "pad_token_id"] = "zero"
    position: Literal["zero", "pad_token_id"] = "zero"
    spatial_position: Literal["zero"] = "zero"
    image: Literal["white", "black", "random", "mean"] = "black"


class SequenceBaselineGenerator(BaselineGenerator[SequenceBaselineGeneratorConfig]):
    __config__ = SequenceBaselineGeneratorConfig

    def __init__(
        self,
        model: TransformersEncoderModel,
        config: SequenceBaselineGeneratorConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        assert isinstance(model, TransformersEncoderModel), (
            "SequenceBaselineGenerator only supports TransformersEncoderModel"
        )
        self._model = model

    @property
    def special_token_ids(self) -> dict[str, int]:
        return self._model.config.embeddings.special_token_ids

    def __call__(  # type: ignore[override]
        self, inputs: torch.Tensor | OrderedDict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor] | torch.Tensor:
        assert isinstance(inputs, OrderedDict), (
            "SequenceBaselineGenerator only supports inputs as OrderedDict"
        )
        logger.debug(
            f"Generating sequence baselines using feature-based generator for inputs: {inputs.keys()}"
        )
        input_embeddings = self._model.ids_to_embeddings(**inputs).to_ordered_dict()
        baseline_embeddings = OrderedDict()
        for key in input_embeddings.keys():
            if input_embeddings[key] is None:
                raise ValueError(f"{key} embeddings cannot be None")

            baseline_type = getattr(self.config, key)
            baseline_embeddings[key] = self._replace_embeddings(
                embeddings=input_embeddings[key],
                baseline_embeddings=self._create_baseline_from_input(
                    baseline_type, input_embeddings[key]
                ),
                special_tokens_mask=self._get_special_tokens_mask(inputs["token_ids"]),
            )

        return baseline_embeddings

    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        import torch

        special_tokens_mask = torch.zeros_like(
            input_ids, dtype=torch.bool, device=input_ids.device
        )
        for token_id in self.special_token_ids.values():
            special_tokens_mask[input_ids == token_id] = True
        return special_tokens_mask

    def _replace_embeddings(
        self, embeddings, baseline_embeddings, special_tokens_mask=None
    ):
        import torch

        batch_size, seq_len, _ = embeddings.size()

        # make a mask to replace non-special tokens
        replace_mask = torch.full((batch_size, seq_len), 1, device=embeddings.device)

        if special_tokens_mask is not None:
            replace_mask.masked_fill_(special_tokens_mask, value=0.0)

        # convert to bool
        replace_mask = replace_mask.bool().unsqueeze(-1).expand_as(embeddings)

        # expand the mask to the same size as the embeddings
        return embeddings * ~replace_mask + replace_mask * baseline_embeddings

    def _create_embedding_from_special_token(
        self,
        input_tensor: torch.Tensor,
        token_id: int,
        target_embedding: str = "token_embedding",
    ) -> torch.Tensor:
        import torch

        embeddings = self._model.ids_to_embeddings(
            torch.full_like(input_tensor, token_id)
        ).to_ordered_dict()
        assert embeddings[target_embedding] is not None, (
            "Token embeddings should not be None"
        )
        return embeddings[target_embedding]

    # get special tokens mask
    def _create_baseline_from_input(
        self,
        baseline_type: Literal["zero", "mask_token_id", "pad_token_id"],
        input_embeddings: torch.Tensor,
    ):
        import torch

        if baseline_type == "zero":
            return torch.zeros_like(input_embeddings)
        elif baseline_type in ["mask_token_id", "pad_token_id"]:
            return self._create_embedding_from_special_token(
                input_embeddings, self.special_token_ids[baseline_type]
            )
        else:
            raise ValueError(
                f"Invalid masking type: {baseline_type} for word embeddings. Supported types are 'zero', 'mask_token', and 'pad_token'"
            )
