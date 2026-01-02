from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

from atria_registry._module_base import ModuleConfig
from pydantic import Field

from atria_insights.feature_segmentors._base import (
    FeatureSegmentor,
    NoOpSegmenterConfig,
)
from atria_insights.feature_segmentors._image import ImageSegmentorConfigType

if TYPE_CHECKING:
    import torch


class SequenceFeatureMaskSegmentorConfig(ModuleConfig):
    module_path: str | None = (
        "atria_insights.feature_segmentors._sequence.SequenceFeatureMaskSegmentor"
    )
    type: Literal["sequence"] = "sequence"
    group_tokens_to_words: bool = True
    image_segmentor: ImageSegmentorConfigType = Field(
        default_factory=lambda: NoOpSegmenterConfig()
    )

    def build(self, **kwargs) -> Any:
        assert "special_token_ids" in kwargs, (
            "special_token_ids must be provided to build SequenceFeatureMaskSegmentor"
        )
        return super().build(**kwargs)


class SequenceFeatureMaskSegmentor(
    FeatureSegmentor[SequenceFeatureMaskSegmentorConfig]
):
    __config__ = SequenceFeatureMaskSegmentorConfig

    def __init__(
        self,
        config: SequenceFeatureMaskSegmentorConfig,
        special_token_ids: dict[str, int | None],
    ) -> None:
        super().__init__(config)
        self._image_segmentor = config.image_segmentor.build()
        self._special_token_ids = special_token_ids

    def _segment_tokens(
        self, token_ids: torch.Tensor, word_ids: list[list[int]]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        import torch

        feature_masks = []
        special_token_ids_batch = []
        if self.config.group_tokens_to_words:
            # here we group all the tokens to their respective words and keep each feature, words, 2d positions, and 1d positions separate
            for word_ids_per_sample, input_ids_per_sample in zip(
                word_ids, token_ids, strict=True
            ):  # iterate over batch of inputs and word ids
                # all tokens belonging to the same word have the same id [None, 0, 0, 1, 1, 1, 2, ...]
                # None represents CLS/SEP/PAD tokens that are not part of any word
                # we replace None with -100 so we can conver it into a tensor
                # which are then later replaced with 0, 1, 2...
                # each word is assigned a unique id starting from 0 and therefore total features become
                # (number of words + 3)
                feature_mask = torch.tensor(
                    [-100 if x is None else x for x in word_ids_per_sample],
                    device=token_ids.device,
                )
                if feature_mask[feature_mask != -100].numel() > 0:
                    min_word_id = feature_mask[feature_mask != -100].min().item()
                else:
                    min_word_id = 0

                # reset the min_word_id to 0
                feature_mask -= min_word_id

                # add a unique id for cls, cls_end, and ref tokens 0, 1, 2... from start for each token
                n_special_tokens = 0
                for token_id in list(self._special_token_ids.values()):
                    if token_id is not None and token_id in input_ids_per_sample:
                        feature_mask += 1
                        feature_mask[input_ids_per_sample == token_id] = 0
                        n_special_tokens += 1
                special_token_ids = torch.tensor(
                    list(range(n_special_tokens)), device=token_ids.device
                )

                special_token_ids_batch.append(special_token_ids)
                feature_masks.append(feature_mask)
        else:
            for word_ids_per_sample, input_ids_per_sample in zip(
                word_ids, token_ids, strict=True
            ):  # iterate over batch of inputs and word ids
                feature_mask = torch.tensor(
                    [-100 if x is None else x for x in word_ids_per_sample],
                    device=token_ids.device,
                )

                # assign 1 - n to each non-cls/pad/sep token
                feature_mask[feature_mask != -100] = torch.arange(
                    feature_mask[feature_mask != -100].shape[0], device=token_ids.device
                )

                # set cls token to 0
                # add a unique id for cls, cls_end, and ref tokens 0, 1, 2... from start for each token
                n_special_tokens = 0
                for token_id in list(self._special_token_ids.values()):
                    if token_id is not None and token_id in input_ids_per_sample:
                        feature_mask += 1
                        feature_mask[input_ids_per_sample == token_id] = 0
                        n_special_tokens += 1
                special_token_ids = torch.tensor(
                    list(range(n_special_tokens)), device=token_ids.device
                )

                # now set all the pad tokens to the pad id so they are all removed the together in case of
                # metric calculation such as infidelity or aopc
                special_token_ids_batch.append(special_token_ids)
                feature_masks.append(feature_mask)
        return torch.stack(feature_masks), special_token_ids_batch

    def _create_token_level_feature_mask(
        self, token_ids: torch.Tensor, word_ids: list[list[int]]
    ) -> tuple[OrderedDict[str, torch.Tensor], list[torch.Tensor]]:
        import torch

        token_feature_mask, special_token_ids_batch = self._segment_tokens(
            token_ids, word_ids
        )

        # create feature masks for each type of embedding
        feature_masks = OrderedDict()
        mask_max = token_feature_mask.clone().max(dim=1, keepdim=True).values
        for idx, key in enumerate(
            ["input", "position", "token_type", "spatial_position"]
        ):
            feature_masks[key] = token_feature_mask.clone() + idx * mask_max + 1

        # remove frozen features
        frozen_features_per_type = {}
        for key in feature_masks.keys():
            for feature_mask, special_token_ids in zip(
                feature_masks[key], special_token_ids_batch, strict=True
            ):
                if key not in frozen_features_per_type:
                    frozen_features_per_type[key] = []
                frozen_features_per_type[key].append(
                    feature_mask.min() + special_token_ids
                )
        frozen_features_per_type = list(frozen_features_per_type.values())
        frozen_features_batch = []
        for n in range(len(frozen_features_per_type[0])):
            frozen_features = []
            for i in range(len(frozen_features_per_type)):
                frozen_features.append(frozen_features_per_type[i][n])
            frozen_features = torch.cat(frozen_features)
            frozen_features_batch.append(frozen_features)

        return feature_masks, frozen_features_batch

    def _prepare_patch_level_feature_masks(
        self, patch_embeddings: torch.Tensor
    ) -> torch.Tensor:
        bsz = patch_embeddings.shape[0]
        return (
            torch.arange(patch_embeddings.shape[1], device=patch_embeddings.device)
            .unsqueeze(0)
            .repeat(bsz, 1)
        )

    def _create_image_level_feature_mask(self, image: torch.Tensor) -> torch.Tensor:
        return self._image_segmentor(image)

    def __call__(  # type: ignore[override]
        self,
        inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
        word_ids: list[list[int]],
    ) -> tuple[OrderedDict[str, torch.Tensor], list[torch.Tensor]]:
        assert isinstance(inputs, OrderedDict), (
            "SequenceFeatureMaskSegmentor expects inputs to be an OrderedDict with keys: "
            "'token_ids', 'word_ids', and optionally 'image'"
        )
        token_ids = inputs["token_ids"]
        image = inputs.get("pixel_values", None)

        token_feature_masks, frozen_features_per_sample = (
            self._create_token_level_feature_mask(token_ids, word_ids)
        )

        if image is not None:
            image_feature_mask = self._create_image_level_feature_mask(image)
            # add image feature mask to all types of embeddings
            return OrderedDict(
                **token_feature_masks, image=image_feature_mask
            ), frozen_features_per_sample
        return OrderedDict(**token_feature_masks), frozen_features_per_sample
