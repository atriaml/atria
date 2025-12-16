from typing import Annotated, Any, Literal

import torch
from atria_registry._module_base import ModuleConfig
from lime.wrappers.scikit_image import SegmentationAlgorithm
from pydantic import Field


class GridSegmenter:
    def __init__(self, cell_size: int = 16) -> None:
        self.cell_size = cell_size

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
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


class FeatureSegmentorConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True
    type: Literal["grid", "quickshift", "felzenszwalb", "slic"]

    def build(  # type: ignore
        self, **kwargs: Any
    ) -> GridSegmenter | SegmentationAlgorithm:
        if self.type == "grid":
            return GridSegmenter(**kwargs)
        else:
            return SegmentationAlgorithm(self.type, **kwargs)


class GridSegmenterConfig(ModuleConfig):
    cell_size: int = 16

    def build(self, **kwargs: Any) -> GridSegmenter:
        return GridSegmenter(cell_size=self.cell_size)


class QuickshiftSegmenterConfig(ModuleConfig):
    type: Literal["quickshift"] = "quickshift"
    kernel_size: int = 4
    max_dist: int = 200
    ratio: float = 0.2

    def build(self, **kwargs: Any) -> SegmentationAlgorithm:
        return SegmentationAlgorithm(
            "quickshift",
            kernel_size=self.kernel_size,
            max_dist=self.max_dist,
            ratio=self.ratio,
        )


class FelzenszwalbSegmenterConfig(ModuleConfig):
    type: Literal["felzenszwalb"] = "felzenszwalb"
    scale: float = 100.0
    sigma: float = 0.5
    min_size: int = 50

    def build(self, **kwargs: Any) -> SegmentationAlgorithm:
        return SegmentationAlgorithm(
            "felzenszwalb", scale=self.scale, sigma=self.sigma, min_size=self.min_size
        )


class SlicSegmenterConfig(ModuleConfig):
    type: Literal["slic"] = "slic"
    n_segments: int = 100
    compactness: float = 10.0
    sigma: float = 1.0

    def build(self, **kwargs: Any) -> SegmentationAlgorithm:
        return SegmentationAlgorithm(
            "slic",
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
        )


FeatureSegmentorConfigType = Annotated[
    GridSegmenterConfig,
    QuickshiftSegmenterConfig,
    FelzenszwalbSegmenterConfig,
    SlicSegmenterConfig,
    Field(discriminator="type"),
]
