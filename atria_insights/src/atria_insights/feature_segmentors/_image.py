from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Annotated, Generic, Literal, TypeVar

from atria_registry._module_base import ModuleConfig
from pydantic import Field

from atria_insights.feature_segmentors._base import (
    FeatureSegmentor,
    NoOpSegmenterConfig,
)

if TYPE_CHECKING:
    import torch


class GridSegmenterConfig(ModuleConfig):
    module_path: str | None = "atria_insights.feature_segmentors._image.GridSegmenter"
    type: Literal["grid"] = "grid"
    cell_size: int = 16


class QuickshiftImageSegmenterConfig(ModuleConfig):
    module_path: str | None = (
        "atria_insights.feature_segmentors._image.QuickshiftImageSegmentor"
    )
    type: Literal["quickshift"] = "quickshift"
    kernel_size: int = 4
    max_dist: int = 200
    ratio: float = 0.2


class FelzenszwalbImageSegmenterConfig(ModuleConfig):
    module_path: str | None = (
        "atria_insights.feature_segmentors._image.FelzenszwalbImageSegmentor"
    )
    type: Literal["felzenszwalb"] = "felzenszwalb"
    scale: float = 100.0
    sigma: float = 0.5
    min_size: int = 50


class SlicImageSegmenterConfig(ModuleConfig):
    module_path: str | None = (
        "atria_insights.feature_segmentors._image.SlicImageSegmentor"
    )
    type: Literal["slic"] = "slic"
    n_segments: int = 100
    compactness: float = 10.0
    sigma: float = 1.0


T_ScikitImageSegmenterConfig = TypeVar(
    "T_ScikitImageSegmenterConfig",
    bound=QuickshiftImageSegmenterConfig
    | FelzenszwalbImageSegmenterConfig
    | SlicImageSegmenterConfig,
)


class GridSegmenter(FeatureSegmentor[GridSegmenterConfig]):
    __config__ = GridSegmenterConfig

    def _image_batch_to_mask(self, image_batch: torch.Tensor) -> torch.Tensor:
        import torch

        assert image_batch.dim() == 4, (
            "Input images must be a 4D tensor (B x C x H x W)"
        )
        feature_mask = []
        for image in image_batch:
            # image dimensions are C x H x H
            dim_x, dim_y = (
                image.shape[1] // self.config.cell_size,
                image.shape[2] // self.config.cell_size,
            )
            mask = (
                torch.arange(dim_x * dim_y, device=image_batch.device)
                .view((dim_x, dim_y))
                .repeat_interleave(self.config.cell_size, dim=0)
                .repeat_interleave(self.config.cell_size, dim=1)
                .long()
                .unsqueeze(0)
            )
            feature_mask.append(mask.expand_as(image))
        return torch.stack(feature_mask)

    def __call__(  # type: ignore[override]
        self, images: torch.Tensor | OrderedDict[str, torch.Tensor]
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        if isinstance(images, OrderedDict):
            return OrderedDict(
                (key, self._image_batch_to_mask(imgs)) for key, imgs in images.items()
            )
        else:
            return self._image_batch_to_mask(images)


class ScikitImageSegmenter(FeatureSegmentor, Generic[T_ScikitImageSegmenterConfig]):
    __abstract__ = True

    def __init__(self, config: T_ScikitImageSegmenterConfig | None = None):
        from lime.wrappers.scikit_image import SegmentationAlgorithm

        super().__init__(config=config)

        kwargs = self.config.kwargs
        algo_type = kwargs.pop("type", None)
        self._segmenter = SegmentationAlgorithm(algo_type=algo_type, **kwargs)

    def _image_batch_to_mask(self, image_batch: torch.Tensor) -> torch.Tensor:
        assert image_batch.dim() == 4, (
            "Input images must be a 4D tensor (B x C x H x W)"
        )
        np_images = image_batch.permute(0, 2, 3, 1).cpu().numpy()
        feature_masks = []
        for np_image in np_images:
            mask = self._segmenter(np_image)
            feature_masks.append(
                torch.tensor(mask, device=image_batch.device).unsqueeze(0)
            )
        return torch.stack(feature_masks)

    def __call__(  # type: ignore[override]
        self, images: torch.Tensor | OrderedDict[str, torch.Tensor]
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        if isinstance(images, OrderedDict):
            return OrderedDict(
                (key, self._image_batch_to_mask(imgs)) for key, imgs in images.items()
            )
        else:
            return self._image_batch_to_mask(images)


class SlicImageSegmentor(ScikitImageSegmenter[SlicImageSegmenterConfig]):
    __config__ = SlicImageSegmenterConfig


class FelzenszwalbImageSegmentor(
    ScikitImageSegmenter[FelzenszwalbImageSegmenterConfig]
):
    __config__ = FelzenszwalbImageSegmenterConfig


class QuickshiftImageSegmentor(ScikitImageSegmenter[QuickshiftImageSegmenterConfig]):
    __config__ = QuickshiftImageSegmenterConfig


ImageSegmentorConfigType = Annotated[
    NoOpSegmenterConfig
    | GridSegmenterConfig
    | QuickshiftImageSegmenterConfig
    | FelzenszwalbImageSegmenterConfig
    | SlicImageSegmenterConfig,
    Field(discriminator="type"),
]
