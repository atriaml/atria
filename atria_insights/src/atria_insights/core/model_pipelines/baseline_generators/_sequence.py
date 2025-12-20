from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    import torch
    from atria_models.core.models.transformers._models._encoder_model import (
        TransformersEncoderModel,
    )


class SequenceBaselinesConfig(BaseModel):
    text: Literal["zero", "mask_token_id", "pad_token_id"] = "zero"
    token_type: Literal["zero", "pad_token_id"] = "zero"
    position: Literal["zero", "pad_token_id"] = "zero"
    spatial_position: Literal["zero"] = "zero"
    image: Literal["white", "black", "random", "mean"] = "black"


class SequenceBaselineGenerator:
    def __init__(
        self,
        *,
        model: TransformersEncoderModel,
        baselines_config: SequenceBaselinesConfig,
    ) -> None:
        self._model = model
        self._baselines_config = baselines_config

    @property
    def special_token_ids(self) -> dict[str, int]:
        return self._model.config.embeddings.special_token_ids

    def __call__(
        self,
        inputs: OrderedDict[str, torch.Tensor],
        target_inputs: list[str] | None = None,
    ) -> OrderedDict[str, torch.Tensor]:
        input_embeddings = self._model.ids_to_embeddings(**inputs).to_ordered_dict()
        baseline_embeddings = OrderedDict()
        for key in input_embeddings.keys():
            if target_inputs is not None and key not in target_inputs:
                continue
            if input_embeddings[key] is None:
                raise ValueError(f"{key} embeddings cannot be None")

            baseline_embeddings[key] = self._replace_embeddings(
                embeddings=input_embeddings[key],
                baseline_embeddings=self._create_baseline_from_input(
                    self._baselines_config.text, input_embeddings[key]
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


class SequenceBaselineGeneratorConfig(BaseModel):
    type: Literal["sequence"] = "sequence"
    baselines_config: SequenceBaselinesConfig = SequenceBaselinesConfig()

    def build(
        self, *, model: TransformersEncoderModel, **kwargs: Any
    ) -> SequenceBaselineGenerator:
        return SequenceBaselineGenerator(
            model=model, baselines_config=self.baselines_config
        )
