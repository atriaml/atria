from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal

from atria_registry._module_base import ModuleConfig
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import torch
    from lime.wrappers.scikit_image import SegmentationAlgorithm


class GridSegmenter:
    def __init__(self, cell_size: int = 16) -> None:
        self.cell_size = cell_size

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert images.dim() == 4, "Input images must be a 4D tensor (B x C x H x W)"
        feature_mask = []
        for image in images:
            # image dimensions are C x H x H
            dim_x, dim_y = (
                image.shape[1] // self.cell_size,
                image.shape[2] // self.cell_size,
            )
            mask = (
                torch.arange(dim_x * dim_y, device=images.device)
                .view((dim_x, dim_y))
                .repeat_interleave(self.cell_size, dim=0)
                .repeat_interleave(self.cell_size, dim=1)
                .long()
                .unsqueeze(0)
            )
            feature_mask.append(mask)
        return torch.stack(feature_mask)


class SequenceFeatureMaskSegmentor:
    def __init__(
        self,
        image_segmentor: Callable,
        group_tokens_to_words: bool = True,
        special_token_ids: dict[str, int] | None = None,
    ) -> None:
        self._image_segmentor = image_segmentor
        self._group_tokens_to_words = group_tokens_to_words
        self._special_token_ids = special_token_ids or {}

    def _segment_tokens(
        self, token_ids: torch.Tensor, word_ids: list[list[int]]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feature_masks = []
        special_token_ids_batch = []
        if self._group_tokens_to_words:
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
        self, input_ids: torch.Tensor, word_ids: list[list[int]]
    ) -> tuple[OrderedDict[str, torch.Tensor], list[torch.Tensor]]:
        token_feature_mask, special_token_ids_batch = self._segment_tokens(
            input_ids, word_ids
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

    def __call__(
        self,
        input_ids: torch.Tensor,
        word_ids: list[list[int]],
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[OrderedDict[str, torch.Tensor], list[torch.Tensor]]:
        token_feature_masks, frozen_features_per_sample = (
            self._create_token_level_feature_mask(input_ids, word_ids)
        )
        if pixel_values is not None:
            image_feature_mask = self._create_image_level_feature_mask(pixel_values)
            # add image feature mask to all types of embeddings
        return OrderedDict(**token_feature_masks, image=image_feature_mask)


class ScikitImageSegmenter:
    def __init__(self, segmenter: SegmentationAlgorithm) -> None:
        self._segmenter = segmenter

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert images.dim() == 4, "Input images must be a 4D tensor (B x C x H x W)"
        np_images = images.permute(0, 2, 3, 1).cpu().numpy()
        feature_masks = []
        for np_image in np_images:
            mask = self._segmenter(np_image)
            feature_masks.append(torch.tensor(mask, device=images.device).unsqueeze(0))
        return torch.stack(feature_masks)


class NoOpSegmenterConfig(ModuleConfig):
    type: Literal["noop"] = "noop"

    def build(self, **kwargs: Any) -> Callable:
        return lambda x: None


class GridSegmenterConfig(ModuleConfig):
    type: Literal["grid"] = "grid"
    cell_size: int = 16

    def build(self, **kwargs: Any) -> GridSegmenter:
        return GridSegmenter(cell_size=self.cell_size)


class QuickshiftSegmenterConfig(ModuleConfig):
    type: Literal["quickshift"] = "quickshift"
    kernel_size: int = 4
    max_dist: int = 200
    ratio: float = 0.2

    def build(self, **kwargs: Any) -> ScikitImageSegmenter:
        from lime.wrappers.scikit_image import SegmentationAlgorithm

        return ScikitImageSegmenter(
            segmenter=SegmentationAlgorithm(
                "quickshift",
                kernel_size=self.kernel_size,
                max_dist=self.max_dist,
                ratio=self.ratio,
            )
        )


class FelzenszwalbSegmenterConfig(ModuleConfig):
    type: Literal["felzenszwalb"] = "felzenszwalb"
    scale: float = 100.0
    sigma: float = 0.5
    min_size: int = 50

    def build(self, **kwargs: Any) -> ScikitImageSegmenter:
        from lime.wrappers.scikit_image import SegmentationAlgorithm

        return ScikitImageSegmenter(
            segmenter=SegmentationAlgorithm(
                "felzenszwalb",
                scale=self.scale,
                sigma=self.sigma,
                min_size=self.min_size,
            )
        )


class SlicSegmenterConfig(ModuleConfig):
    type: Literal["slic"] = "slic"
    n_segments: int = 100
    compactness: float = 10.0
    sigma: float = 1.0

    def build(self, **kwargs: Any) -> ScikitImageSegmenter:
        from lime.wrappers.scikit_image import SegmentationAlgorithm

        return ScikitImageSegmenter(
            segmenter=SegmentationAlgorithm(
                "slic",
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
            )
        )


ImageSegmentorConfigType = Annotated[
    NoOpSegmenterConfig
    | GridSegmenterConfig
    | QuickshiftSegmenterConfig
    | FelzenszwalbSegmenterConfig
    | SlicSegmenterConfig,
    Field(discriminator="type"),
]


class SequenceFeatureMaskSegmentorConfig(BaseModel):
    type: Literal["sequence"] = "sequence"
    group_tokens_to_words: bool = True

    image_segmentor: ImageSegmentorConfigType = Field(
        default_factory=lambda: NoOpSegmenterConfig()
    )

    def build(
        self, special_token_ids: dict[str, int] | None = None, **kwargs: Any
    ) -> SequenceFeatureMaskSegmentor:
        group_tokens_to_words = kwargs.pop(
            "group_tokens_to_words", self.group_tokens_to_words
        )
        return SequenceFeatureMaskSegmentor(
            image_segmentor=self.image_segmentor.build(),
            group_tokens_to_words=group_tokens_to_words,
            special_token_ids=special_token_ids,
        )


FeatureSegmentorConfigType = Annotated[
    ImageSegmentorConfigType | SequenceFeatureMaskSegmentorConfig,
    Field(discriminator="type"),
]
