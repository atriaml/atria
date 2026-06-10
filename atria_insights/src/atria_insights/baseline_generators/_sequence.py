from __future__ import annotations

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
    __schema_exclude__ = {"image_mean", "image_std"}
    type: Literal["sequence"] = "sequence"
    token_ids: Literal["zero", "mask_token_id", "pad_token_id", "none"] = "zero"
    token_type_ids: Literal["zero", "pad_token_id", "none"] = "zero"
    position_ids: Literal["zero", "pad_token_id", "none"] = "zero"
    layout_ids: Literal["zero", "pad_token_id", "none"] = "zero"
    image: Literal["white", "black", "random", "mean", "none"] = "black"
    image_mean: list[float] | None = None
    image_std: list[float] | None = None

    @classmethod
    def baseline_types_per_modality(cls) -> dict[str, list[str]]:
        return {
            "token_ids": ["zero", "mask_token_id", "pad_token_id", "none"],
            "token_type_ids": ["zero", "pad_token_id", "none"],
            "position_ids": ["zero", "pad_token_id", "none"],
            "layout_ids": ["zero", "pad_token_id", "none"],
            "image": ["white", "black", "random", "mean", "none"],
        }


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
        self._model: TransformersEncoderModel

    @property
    def special_token_ids(self) -> dict[str, int | None]:
        return self._model.config.embeddings_config.special_token_ids

    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        import torch

        special_tokens_mask = torch.zeros_like(
            input_ids, dtype=torch.bool, device=input_ids.device
        )
        for token_id in self.special_token_ids.values():
            special_tokens_mask[input_ids == token_id] = True
        return special_tokens_mask

    def _replace_embedding(
        self,
        embedding: torch.Tensor,
        baseline_embedding: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
        masking_probability=1.0,
    ):
        import torch

        batch_size, seq_len, _ = embedding.size()

        # generate the mask probability matrix for masking
        probability_matrix = torch.full(
            (batch_size, seq_len), masking_probability, device=embedding.device
        )

        # set the mask probability of special tokens to be 0
        if special_tokens_mask is not None:
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # generate the replacement mask matrix
        replacement_mask = (
            torch.bernoulli(probability_matrix)
            .bool()
            .unsqueeze(-1)
            .expand_as(embedding)
        )

        # expand the mask to the same size as the embeddings
        return embedding * ~replacement_mask + replacement_mask * baseline_embedding

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
            return normalize(torch.rand_like(image), mean=mean, std=std)
        elif baseline_type == "mean":
            # mean lies at 0 after normalization
            return torch.zeros_like(image)
        else:
            raise ValueError(
                f"Invalid image baseline type: {baseline_type}. Supported types are 'white', 'black', 'random', and 'mean'"
            )

    def _get_sequence_baselines(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        import torch

        baseline_input_ids = {}
        zero_baselines = {}
        for input_key, input_ids in inputs.items():
            baseline_type = getattr(self.config, input_key)
            if baseline_type == "none":
                baseline_input_ids[input_key] = input_ids
                zero_baselines[input_key] = False
                continue
            elif baseline_type == "zero":
                baseline_input_ids[input_key] = input_ids
                zero_baselines[input_key] = True
            elif baseline_type in ["mask_token_id", "pad_token_id"]:
                baseline_token_id = self.special_token_ids[baseline_type]
                if input_key in ["layout_ids", "token_type_ids"]:
                    assert baseline_type == "pad_token_id", (
                        f"Only 'pad_token_id' is supported for {input_key}"
                    )
                    baseline_token_id = (
                        0  # for layout_ids and token_type_ids, pad_token_id is 0
                    )
                assert baseline_token_id is not None, (
                    f"{baseline_type} is not defined in the model's special token ids"
                )
                baseline_input_ids[input_key] = torch.full_like(
                    input_ids, baseline_token_id
                )
                zero_baselines[input_key] = False
        baseline_ids_to_embeddings = self._model.ids_to_embeddings(
            **baseline_input_ids
        ).to_id_map()
        for key in baseline_ids_to_embeddings.keys():
            if zero_baselines[key]:
                baseline_ids_to_embeddings[key] = torch.zeros_like(
                    baseline_ids_to_embeddings[key]
                )
        return baseline_ids_to_embeddings

    def __call__(  # type: ignore[override]
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        assert isinstance(inputs, dict), (
            "SequenceBaselineGenerator only supports inputs as dict"
        )
        logger.debug(
            f"Generating sequence baselines using feature-based generator for inputs: {inputs.keys()}"
        )
        assert "token_ids" in inputs, "token_ids must be provided in the inputs"
        sequence_inputs = {k: v for k, v in inputs.items() if k != "image"}
        input_ids_to_embeddings = self._model.ids_to_embeddings(
            **sequence_inputs
        ).to_id_map()
        baseline_embeddings = self._get_sequence_baselines(sequence_inputs)
        input_keys = list(input_ids_to_embeddings.keys())

        # now go over each embedding type and create baselines by replacing input embeddings
        # with baselines while keeping special tokens unchanged
        for input_key in input_keys:
            baseline_embeddings[input_key] = self._replace_embedding(
                embedding=input_ids_to_embeddings[input_key],
                baseline_embedding=baseline_embeddings[input_key],
                special_tokens_mask=self._get_special_tokens_mask(
                    sequence_inputs["token_ids"]
                ),
            )

        image = inputs.get("image", None)
        if image is not None:
            if self.config.image == "none":
                baseline_embeddings["image"] = image
            else:
                # assert (
                #     self.config.image_mean is not None
                #     and self.config.image_std is not None
                # ), (
                #     "image_mean and image_std must be provided for image baseline generation"
                # )
                # validate shape of image
                assert image.ndim == 4, (
                    f"Image input should be 4-dimensional (B, C, H, W), but got {sequence_inputs['image'].ndim} dimensions"
                )
                baseline_embeddings["image"] = self._create_image_baseline(
                    image=image,
                    baseline_type=self.config.image,
                    mean=self.config.image_mean or [0.5 ,0.5, 0.5],
                    std=self.config.image_std or [0.5 ,0.5, 0.5],
                )
        return baseline_embeddings


class NoEmbedSequenceBaselineGeneratorConfig(SequenceBaselineGeneratorConfig):
    type: Literal["no_embed_sequence"] = "no_embed_sequence"


class NoEmbedSequenceBaselineGenerator(
    BaselineGenerator[NoEmbedSequenceBaselineGeneratorConfig]
):
    __config__ = NoEmbedSequenceBaselineGeneratorConfig

    def __init__(
        self,
        model: TransformersEncoderModel,
        config: NoEmbedSequenceBaselineGeneratorConfig | None = None,
    ) -> None:
        from atria_models.core.models.transformers._models._encoder_model import (
            TransformersEncoderModel,
        )

        super().__init__(model=model, config=config)
        assert isinstance(self._model, TransformersEncoderModel), (
            "NoEmbedSequenceBaselineGenerator only supports TransformersEncoderModel"
        )
        self._model: TransformersEncoderModel

    @property
    def special_token_ids(self) -> dict[str, int | None]:
        return self._model.config.embeddings_config.special_token_ids

    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        import torch

        special_tokens_mask = torch.zeros_like(
            input_ids, dtype=torch.bool, device=input_ids.device
        )
        for token_id in self.special_token_ids.values():
            special_tokens_mask[input_ids == token_id] = True
        return special_tokens_mask

    def _replace_inputs(
        self,
        input: torch.Tensor,
        baseline_input: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
        masking_probability=1.0,
    ):
        import torch

        batch_size, seq_len = input.size()[:2]

        # generate the mask probability matrix for masking
        probability_matrix = torch.full(
            (batch_size, seq_len), masking_probability, device=input.device
        )

        # set the mask probability of special tokens to be 0
        if special_tokens_mask is not None:
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # generate the replacement mask matrix
        if len(input.size()) == 3:
            replacement_mask = (
                torch.bernoulli(probability_matrix)
                .bool()
                .unsqueeze(-1)
                .expand_as(input)
            )
        else:
            replacement_mask = torch.bernoulli(probability_matrix).bool()

        # expand the mask to the same size as the embeddings
        return input * ~replacement_mask + replacement_mask * baseline_input

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
            return normalize(torch.rand_like(image), mean=mean, std=std)
        elif baseline_type == "mean":
            # mean lies at 0 after normalization
            return torch.zeros_like(image)
        else:
            raise ValueError(
                f"Invalid image baseline type: {baseline_type}. Supported types are 'white', 'black', 'random', and 'mean'"
            )

    def _get_sequence_baselines(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        import torch

        baseline_input_ids = {}
        zero_baselines = {}
        for input_key, input_ids in inputs.items():
            baseline_type = getattr(self.config, input_key)
            if baseline_type == "none":
                baseline_input_ids[input_key] = input_ids
                zero_baselines[input_key] = False
                continue
            elif baseline_type == "zero":
                baseline_input_ids[input_key] = input_ids
                zero_baselines[input_key] = True
            elif baseline_type in ["mask_token_id", "pad_token_id"]:
                baseline_token_id = self.special_token_ids[baseline_type]
                if input_key in ["layout_ids", "token_type_ids"]:
                    assert baseline_type == "pad_token_id", (
                        f"Only 'pad_token_id' is supported for {input_key}"
                    )
                    baseline_token_id = (
                        0  # for layout_ids and token_type_ids, pad_token_id is 0
                    )
                assert baseline_token_id is not None, (
                    f"{baseline_type} is not defined in the model's special token ids"
                )
                baseline_input_ids[input_key] = torch.full_like(
                    input_ids, baseline_token_id
                )
                zero_baselines[input_key] = False

        for key in baseline_input_ids.keys():
            if zero_baselines[key]:
                baseline_input_ids[key] = torch.zeros_like(baseline_input_ids[key])
        return baseline_input_ids

    def __call__(  # type: ignore[override]
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        assert isinstance(inputs, dict), (
            "SequenceBaselineGenerator only supports inputs as dict"
        )
        logger.debug(
            f"Generating sequence baselines using feature-based generator for inputs: {inputs.keys()}"
        )
        assert "token_ids" in inputs, "token_ids must be provided in the inputs"
        sequence_inputs = {k: v for k, v in inputs.items() if k != "image"}
        baseline_inputs = self._get_sequence_baselines(sequence_inputs)
        input_keys = list(sequence_inputs.keys())

        # now go over each embedding type and create baselines by replacing input embeddings
        # with baselines while keeping special tokens unchanged
        for input_key in input_keys:
            baseline_inputs[input_key] = self._replace_inputs(
                input=sequence_inputs[input_key],
                baseline_input=baseline_inputs[input_key],
                special_tokens_mask=self._get_special_tokens_mask(
                    sequence_inputs["token_ids"]
                ),
            )

        image = inputs.get("image", None)
        if image is not None:
            if self.config.image == "none":
                baseline_inputs["image"] = image
            else:
                assert (
                    self.config.image_mean is not None
                    and self.config.image_std is not None
                ), (
                    "image_mean and image_std must be provided for image baseline generation"
                )
                # validate shape of image
                assert image.ndim == 4, (
                    f"Image input should be 4-dimensional (B, C, H, W), but got {sequence_inputs['image'].ndim} dimensions"
                )
                baseline_inputs["image"] = self._create_image_baseline(
                    image=image,
                    baseline_type=self.config.image,
                    mean=self.config.image_mean,
                    std=self.config.image_std,
                )
        return baseline_inputs
