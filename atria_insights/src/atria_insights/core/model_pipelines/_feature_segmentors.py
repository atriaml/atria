from collections.abc import Callable
from typing import Annotated, Any, Literal

import torch
from atria_registry._module_base import ModuleConfig
from lime.wrappers.scikit_image import SegmentationAlgorithm
from pydantic import Field


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
    def build(self, **kwargs: Any) -> Callable:
        return lambda x: x


class GridSegmenterConfig(ModuleConfig):
    cell_size: int = 16

    def build(self, **kwargs: Any) -> GridSegmenter:
        return GridSegmenter(cell_size=self.cell_size)


class QuickshiftSegmenterConfig(ModuleConfig):
    type: Literal["quickshift"] = "quickshift"
    kernel_size: int = 4
    max_dist: int = 200
    ratio: float = 0.2

    def build(self, **kwargs: Any) -> ScikitImageSegmenter:
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
        return ScikitImageSegmenter(
            segmenter=SegmentationAlgorithm(
                "slic",
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
            )
        )


FeatureSegmentorConfigType = Annotated[
    NoOpSegmenterConfig,
    GridSegmenterConfig,
    QuickshiftSegmenterConfig,
    FelzenszwalbSegmenterConfig,
    SlicSegmenterConfig,
    Field(discriminator="type"),
]
