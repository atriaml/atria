from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
from atria_registry import ModuleConfig

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

class SequenceBaselineGenerator:
    def __init__(self, model_adaptor: ModelAdaptor) -> None:
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        if position_ids is None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = (
                self._model.layoutlmv3.embeddings.create_position_ids_from_input_ids(
                    input_ids, self._model.layoutlmv3.embeddings.padding_idx
                ).to(input_ids.device)
            )

        return OrderedDict(
            input_embeddings=self._prepare_input_ids_baselines(
                input_ids, self._baselines_config
            ),
            position_embeddings=self._prepare_position_ids_baselines(
                input_ids, position_ids, self._baselines_config
            ),
            spatial_position_embeddings=self._generate_spatial_position_baselines(
                input_ids, bbox, self._baselines_config
            ),
            # notice here that the patch embeddings are generated from the pixel values first and then
            # the patch embeddings baselines are generated according to the patch embeddings shape
            patch_embeddings=self._generate_patch_embedding_baselines(
                pixel_values, self._baselines_config
            ),
        )

    def _prepare_input_ids_baselines(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_ids_baselines = (
            attention_mask * self.pad_token_id
            + (1 - attention_mask) * self.pad_token_id
        )
        input_ids_baselines[input_ids == self.bos_token_id] = self.bos_token_id
        input_ids_baselines[input_ids == self.eos_token_id] = self.eos_token_id
        return input_ids_baselines


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
