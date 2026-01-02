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
    token_ids: Literal["zero", "mask_token_id", "pad_token_id"] = "zero"
    token_type_ids: Literal["zero", "pad_token_id"] = "zero"
    position_ids: Literal["zero", "pad_token_id"] = "zero"
    layout_ids: Literal["zero", "pad_token_id"] = "zero"
    image: Literal["white", "black", "random", "mean"] = "black"
    image_mean: list[float] | None = None
    image_std: list[float] | None = None


class SequenceBaselineGenerator(BaselineGenerator[SequenceBaselineGeneratorConfig]):
    __config__ = SequenceBaselineGeneratorConfig

    def __init__(
        self,
        model: TransformersEncoderModel,
        config: SequenceBaselineGeneratorConfig | None = None,
    ) -> None:
        from atria_models.core.models.transformers._models._encoder_model import (
            TransformersEncoderModel,
        )

        super().__init__(model=model, config=config)
        assert isinstance(self._model, TransformersEncoderModel), (
            "SequenceBaselineGenerator only supports TransformersEncoderModel"
        )

    @property
    def special_token_ids(self) -> dict[str, int]:
        return self._model.config.embeddings_config.special_token_ids

    def __call__(  # type: ignore[override]
        self, inputs: OrderedDict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor] | torch.Tensor:
        assert isinstance(inputs, OrderedDict), (
            "SequenceBaselineGenerator only supports inputs as OrderedDict"
        )
        logger.debug(
            f"Generating sequence baselines using feature-based generator for inputs: {inputs.keys()}"
        )

        image = inputs.pop("image", None)
        baseline_embeddings = OrderedDict()
        ids_to_embeddings = self._model.ids_to_embeddings(**inputs).id_map()
        for key in inputs.keys():
            if ids_to_embeddings[key] is None:
                raise ValueError(f"{key} embeddings cannot be None")

            if key in ["token_ids", "token_type_ids", "position_ids", "layout_ids"]:
                baseline_type = getattr(self.config, key)
                baseline_embeddings[key] = self._replace_embeddings(
                    embeddings=ids_to_embeddings[key],
                    baseline_embeddings=self._create_baseline_from_input(
                        baseline_type, ids_to_embeddings[key], key=key
                    ),
                    special_tokens_mask=self._get_special_tokens_mask(
                        inputs["token_ids"]
                    ),
                )

        if image is not None:
            assert (
                self.config.image_mean is not None and self.config.image_std is not None
            ), "image_mean and image_std must be provided for image baseline generation"
            baseline_embeddings["image"] = self._create_image_baseline(
                image=image,
                baseline_type=self.config.image,
                mean=self.config.image_mean,
                std=self.config.image_std,
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

    # get special tokens mask
    def _create_baseline_from_input(
        self,
        baseline_type: Literal["zero", "mask_token_id", "pad_token_id"],
        input_embeddings: torch.Tensor,
        key: str,
    ):
        import torch

        if baseline_type == "zero":
            return torch.zeros_like(input_embeddings)
        elif baseline_type in ["mask_token_id", "pad_token_id"]:
            token_id = self.special_token_ids[baseline_type]
            if key == "layout_ids":
                # for spatial position embeddings, pad token id is usually 0
                # Note: this is 0 padding
                token_id = 0
            baseline_embeddings = self._model.ids_to_embeddings(
                torch.full_like(input_embeddings, token_id)
            ).to_ordered_dict()
            embedding_key = key.replace("_id", "_embeddings")
            assert baseline_embeddings[embedding_key] is not None, (
                "Token embeddings should not be None"
            )
            return baseline_embeddings[embedding_key]
        else:
            raise ValueError(
                f"Invalid masking type: {baseline_type} for word embeddings. Supported types are 'zero', 'mask_token', and 'pad_token'"
            )

    def _create_image_baseline(
        self,
        image: torch.Tensor,
        baseline_type: Literal["white", "black", "random", "mean"],
        mean: list[float],
        std: list[float],
    ) -> torch.Tensor:
        import torch
        from torchvision.transforms.functional import normalize

        if baseline_type == "white":
            return normalize(torch.ones_like(image), mean=mean, std=std)
        elif baseline_type == "black":
            return normalize(torch.zeros_like(image), mean=mean, std=std)
        elif baseline_type == "random":
            return torch.rand_like(image)
        elif baseline_type == "mean":
            # mean lies at 0 after normalization
            return torch.zeros_like(image)
        else:
            raise ValueError(
                f"Invalid image baseline type: {baseline_type}. Supported types are 'white', 'black', 'random', and 'mean'"
            )
