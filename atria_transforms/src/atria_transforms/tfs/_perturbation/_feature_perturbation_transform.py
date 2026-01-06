from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from atria_insights.baseline_generators._sequence import SequenceBaselineGeneratorConfig
from atria_insights.feature_segmentors._sequence import (
    SequenceFeatureMaskSegmentorConfig,
)
from atria_logger import get_logger
from atria_models.core.model_pipelines.utilities import log_tensors_debug_info
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from pydantic import Field

from atria_transforms.core import DataTransform
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_transforms.registry import DATA_TRANSFORMS

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


@DATA_TRANSFORMS.register("feature_perturbation/sequence")
class SequenceFeaturePertubationTransform(DataTransform[DocumentTensorDataModel]):
    feature_segmentor: SequenceFeatureMaskSegmentorConfig = (
        SequenceFeatureMaskSegmentorConfig()
    )
    baseline_generator: SequenceBaselineGeneratorConfig = (
        SequenceBaselineGeneratorConfig()
    )
    percent_features_perturbed: float = 0.5
    ignored_feature_ids: list[str] = Field(default_factory=list)

    def build(self, model: TransformersEncoderModel) -> None:
        self._model = model
        self._feature_segmentor = self.feature_segmentor.build(
            special_token_ids=model.config.embeddings_config.special_token_ids
        )
        self._baseline_generator = self.baseline_generator.build(model=model)

        # get model ids_to_embeddings args
        self._model_args = inspect.signature(model.forward).parameters.keys()
        self._model_id_to_embeddings_args = inspect.signature(
            model.ids_to_embeddings
        ).parameters.keys()

    def _prepare_inputs(self, batch: DocumentTensorDataModel) -> dict[str, Any]:
        # get default ids
        default_ids = self._model.get_default_ids_from_token_ids(batch.token_ids)
        token_type_ids = (
            batch.token_type_ids
            if batch.token_type_ids is not None
            else default_ids.get("token_type_ids")
        )
        position_ids = default_ids.get("position_ids")
        assert token_type_ids is not None, "Token type ids cannot be None"
        assert position_ids is not None, "Position ids cannot be None"

        # first we generate the embeddings
        inputs = {
            "token_ids": batch.token_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

        is_image_required = (
            "image" in self._model_args or "image" in self._model_id_to_embeddings_args
        )
        if batch.image is not None and is_image_required:
            inputs["image"] = batch.image

        is_bbox_required = (
            "layout_ids" in self._model_args
            or "layout_ids" in self._model_id_to_embeddings_args
        )
        if batch.token_bboxes is not None and is_bbox_required:
            token_bboxes = batch.token_bboxes
            if batch.metadata.bbox_normalized[0]:
                token_bboxes = (batch.token_bboxes * 1000.0).clip(0, 1000).long()
            inputs["layout_ids"] = token_bboxes
        return inputs

    def _prepare_sequence_feature_keys(
        self, explained_inputs: dict[str, torch.Tensor]
    ) -> list[str]:
        possible_feature_keys = []
        for key in ["token_ids", "position_ids", "layout_ids", "token_type_ids"]:
            if key in explained_inputs and key not in self.ignored_feature_ids:
                possible_feature_keys.append(key)
        return possible_feature_keys

    def _perturb_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        baselines: dict[str, torch.Tensor],
        feature_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        import torch

        device = next(iter(inputs.values())).device

        # if no feature masks provided, treat each element as its own feature
        if feature_masks is None:
            # each element is its own feature
            feature_masks = {
                k: torch.arange(v.numel(), device=device).reshape(v.shape)
                for k, v in inputs.items()
            }

        # batch size
        perturbed_inputs = {}
        for feature_key in inputs.keys():
            inputs_per_feature = inputs[feature_key]
            baselines_per_feature = baselines[feature_key]
            feature_masks_per_feature = feature_masks[feature_key]

            perturbation_masks = []
            for feature_mask_tensor in feature_masks_per_feature:
                feature_indices = torch.unique(feature_mask_tensor)
                rand_indices = torch.randperm(len(feature_indices), device=device)[
                    : int(self.percent_features_perturbed * len(feature_indices))
                ]
                rand_perturbation_mask = torch.isin(
                    feature_mask_tensor, feature_indices[rand_indices]
                )
                perturbation_masks.append(rand_perturbation_mask)
            rand_perturbation_mask = torch.stack(perturbation_masks, dim=0)
            perturbed_inputs[feature_key] = (
                inputs_per_feature * (~rand_perturbation_mask)
                + baselines_per_feature * rand_perturbation_mask
            )

        return perturbed_inputs

    def _expand_feature_mask(
        self, feature_mask: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        expanded_feature_mask = {}
        for input_key, input_value in inputs.items():
            mask = feature_mask[input_key]

            # unsqueeze dims to match input shape
            while len(mask.shape) < len(inputs[input_key].shape):
                mask = mask.unsqueeze(-1)

            # expand to match input shape
            mask = mask.expand_as(input_value)

            # assert the shapes match
            assert mask.shape == input_value.shape, (
                f"Feature mask shape {mask.shape} does not match input shape {input_value.shape} for key {input_key}"
            )
            expanded_feature_mask[input_key] = mask
        return expanded_feature_mask

    def _validate_baselines(
        self, inputs: dict[str, torch.Tensor], baselines: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # ensure baselines have the same shape as inputs
        validated_baselines = {}
        for key, value in inputs.items():
            if key not in baselines:
                raise ValueError(f"Baseline for key {key} not found")
            baseline_value = baselines[key]
            if baseline_value.shape != value.shape:
                raise ValueError(
                    f"Baseline shape {baseline_value.shape} does not match input shape {value.shape} for key {key}"
                )
            validated_baselines[key] = baseline_value
        return validated_baselines

    def __call__(self, input: DocumentTensorDataModel) -> DocumentTensorDataModel:
        import torch

        if not isinstance(input, DocumentTensorDataModel):
            raise TypeError(
                f"FeaturePertubationTransform only supports DocumentTensorDataModel, got {type(input)}"
            )

        with torch.no_grad():
            # inputs to embeddings
            inputs = self._prepare_inputs(input)

            # log debug info
            log_tensors_debug_info(inputs, title="inputs")

            # prepare baselines
            baselines = self._baseline_generator(inputs)

            log_tensors_debug_info(baselines, title="baselines")

            # prepare feature mask
            feature_mask, _ = self._feature_segmentor(
                token_ids=inputs["token_ids"],
                image=inputs.get("image", None),
                word_ids=input.word_ids,
                sequence_feature_keys=self._prepare_sequence_feature_keys(inputs),
            )

            # map inputs to embeddings
            input_embeddings = self._model.ids_to_embeddings(
                **{
                    key: inputs[key]
                    for key in self._model_id_to_embeddings_args
                    if key in inputs
                }
            ).to_id_map()

            # filter out ignored feature ids from input embeddings and add them to additional forward kwargs
            ignored_inputs = {}
            for key in self.ignored_feature_ids:
                if key in input_embeddings:
                    ignored_inputs[key] = input_embeddings.pop(key)
                    inputs.pop(key)

            # update inputs
            inputs.update(input_embeddings)

            # validate input shapes
            baselines = self._validate_baselines(inputs=inputs, baselines=baselines)

            # expand feature mask to match input shapes
            feature_mask = self._expand_feature_mask(
                feature_mask=feature_mask, inputs=inputs
            )

            # perturb the inputs
            perturbed_inputs = self._perturb_inputs(
                inputs=inputs, baselines=baselines, feature_masks=feature_mask
            )

            if "layout_ids" in perturbed_inputs:
                perturbed_inputs["layout_embeddings"] = perturbed_inputs.pop(
                    "layout_ids"
                )

            batch_size = len(input.metadata.sample_id)
            return DocumentTensorDataModel(
                **{
                    **input.model_dump(),
                    **perturbed_inputs,
                    **ignored_inputs,
                    "is_embedding": [True] * batch_size,
                    "is_batched": True,
                }
            )
