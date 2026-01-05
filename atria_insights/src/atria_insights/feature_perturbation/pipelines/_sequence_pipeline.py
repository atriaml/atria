from __future__ import annotations

from collections import OrderedDict
from typing import Any, TypeVar

import torch
from atria_insights.feature_perturbation.evaluator_pipelines._config import (
    FeaturePerturbationEvaluatorPipelineConfig,
)
from atria_insights.fp_eval.evaluator_pipelines._base_pipeline import (
    FeaturePerturbationEvaluatorPipeline,
)
from atria_logger import get_logger
from atria_models.core.model_pipelines._sequence_pipeline import (
    QuestionAnsweringPipelineConfig,
    SequenceClassificationPipeline,
    SequenceClassificationPipelineConfig,
    SequenceModelPipeline,
    TokenClassificationPipelineConfig,
)
from atria_models.core.model_pipelines.utilities import log_tensors_debug_info
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_models.core.models.transformers._outputs import (
    TransformersEncoderModelOutput,
)
from atria_models.core.types.model_outputs import ModelOutput
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_types._datasets import DatasetLabels
from pydantic import model_validator

from atria_insights.baseline_generators import BaselineGeneratorConfigType
from atria_insights.baseline_generators._sequence import SequenceBaselineGeneratorConfig
from atria_insights.feature_perturbation._registry_groups import (
    FEATURE_PERTURBATION_EVALUATOR_PIPELINES,
)
from atria_insights.feature_segmentors import FeatureSegmentorConfigType
from atria_insights.feature_segmentors._sequence import (
    SequenceFeatureMaskSegmentorConfig,
)

logger = get_logger(__name__)


class SequenceFeaturePerturbationEvaluatorPipelineConfig(
    FeaturePerturbationEvaluatorPipelineConfig
):
    feature_segmentor: FeatureSegmentorConfigType = SequenceFeatureMaskSegmentorConfig()
    baseline_generator: BaselineGeneratorConfigType = SequenceBaselineGeneratorConfig()

    @model_validator(mode="after")
    def validate_configs(self) -> SequenceFeaturePerturbationEvaluatorPipelineConfig:
        if not isinstance(self.feature_segmentor, SequenceFeatureMaskSegmentorConfig):
            raise ValueError(
                "feature_segmentor must be an instance of SequenceFeatureMaskSegmentorConfig"
            )
        if not isinstance(self.baseline_generator, SequenceBaselineGeneratorConfig):
            raise ValueError(
                "baseline_generator must be an instance of SequenceBaselineGeneratorConfig"
            )
        return self


T_FeaturePerturbationEvaluatorPipelineConfig = TypeVar(
    "T_FeaturePerturbationEvaluatorPipelineConfig",
    bound="FeaturePerturbationEvaluatorPipelineConfig",
)


class SequenceFeaturePerturbationEvaluatorPipeline(
    FeaturePerturbationEvaluatorPipeline[
        T_FeaturePerturbationEvaluatorPipelineConfig, DocumentTensorDataModel
    ]
):
    __abstract__ = True

    def __init__(
        self,
        config: T_FeaturePerturbationEvaluatorPipelineConfig,
        labels: DatasetLabels,
        persist_to_disk: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            labels=labels,
            persist_to_disk=persist_to_disk,
            cache_dir=cache_dir,
        )
        assert isinstance(self._model_pipeline, SequenceModelPipeline), (
            f"{self.__class__.__name__} can only be used with SequenceModelPipeline. Found {self._model_pipeline=}"
        )
        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)

    def _build_feature_segmentor(self):
        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)
        self._feature_segmentor = self.config.feature_segmentor.build(
            special_token_ids=self._model_pipeline._model.config.embeddings_config.special_token_ids
        )

    def _build_baseline_generator(self):
        if isinstance(self.config.baseline_generator, SequenceBaselineGeneratorConfig):
            self._baseline_generator = self.config.baseline_generator.build(
                model=self._model_pipeline._model
            )
        else:
            self._baseline_generator = self.config.baseline_generator.build()

    def _generate_sequence_ids_to_embeddings(
        self, batch: DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        # get default ids
        default_ids = self._model_pipeline._model.get_default_ids_from_token_ids(
            batch.token_ids
        )
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

        if self._model_pipeline.config.use_image and (
            "image" in self._model_signature.parameters.keys()
            or "image" in self._model_id_to_embeddings_inputs_list
        ):
            assert batch.image is not None, "Image cannot be None"
            inputs["image"] = batch.image

        if self._model_pipeline.config.use_bbox and (
            "layout_ids" in self._model_signature.parameters.keys()
            or "layout_ids" in self._model_id_to_embeddings_inputs_list
        ):
            assert batch.token_bboxes is not None, "Token bboxes cannot be None"
            token_bboxes = batch.token_bboxes
            if batch.metadata.bbox_normalized[0] and token_bboxes is not None:
                token_bboxes = (
                    (token_bboxes * 1000.0).clip(0, 1000).long()
                    if token_bboxes is not None
                    else None
                )

            inputs["layout_ids"] = token_bboxes
        return inputs

    def _perturbed_inputs(  # type: ignore[override]
        self, batch: DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)

        explained_inputs = self._generate_sequence_ids_to_embeddings(batch)
        if (
            self._model_pipeline.config.use_image
            and "image" not in explained_inputs
            and "image" in self._model_signature.parameters.keys()
        ):
            assert batch.image is not None, "Image cannot be None"
            explained_inputs["image"] = batch.image
        return explained_inputs

    def _additional_forward_kwargs(self, batch: DocumentTensorDataModel):  # type: ignore[override]
        additional_forward_kwargs = {"attention_mask": batch.attention_mask}
        # for models where the additional wrapped args are not used, we still need to pass them as 'additional_forward_args'
        # and we later filter them out in the wrapped model forward
        if (
            "layout_ids" in self._model_signature.parameters.keys()
            and self._model_pipeline.config.use_bbox
        ):
            assert batch.token_bboxes is not None, "Token bboxes cannot be None"
            token_bboxes = batch.token_bboxes
            if batch.metadata.bbox_normalized[0] and token_bboxes is not None:
                token_bboxes = (
                    (token_bboxes * 1000.0).clip(0, 1000).long()
                    if token_bboxes is not None
                    else None
                )

            additional_forward_kwargs["layout_ids"] = token_bboxes
        return additional_forward_kwargs

    def _baselines(  # type: ignore[override]
        self, explained_inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Generate baselines for the explainer."""
        logger.debug(
            "Generating baselines using baseline generator with config: %s",
            self.config.baseline_generator,
        )
        baselines = self._baseline_generator(explained_inputs, **kwargs)

        # filter out ignored feature ids from baselines
        baselines = {
            k: v
            for k, v in baselines.items()
            if k not in self.config.ignored_feature_ids
        }

        return baselines

    def _feature_mask(  # type: ignore[override]
        self,
        explained_inputs: dict[str, torch.Tensor],
        word_ids: torch.Tensor,
        sequence_feature_keys: list[str],
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """Generate feature mask using the feature segmentor."""
        logger.debug(
            "Generating feature mask using feature segmentor with config: %s",
            self.config.feature_segmentor,
        )
        feature_masks, frozen_features = self._feature_segmentor(
            token_ids=explained_inputs["token_ids"],
            image=explained_inputs.get("image", None),
            word_ids=word_ids,
            sequence_feature_keys=sequence_feature_keys,
        )
        return feature_masks, frozen_features

    def _prepare_sequence_feature_keys(
        self, explained_inputs: dict[str, torch.Tensor]
    ) -> list[str]:
        possible_feature_keys = []
        for key in ["token_ids", "position_ids", "layout_ids", "token_type_ids"]:
            if key in explained_inputs and key not in self.config.ignored_feature_ids:
                possible_feature_keys.append(key)
        return possible_feature_keys

    def _get_loss(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, TransformersEncoderModelOutput):
            assert model_output.head_output is not None, "Head output cannot be None"
            assert model_output.head_output.loss is not None, "Loss cannot be None"
            return model_output.head_output.loss
        raise ValueError("Unsupported model output type for loss extraction")

    def perturbation_robustness_eval_step(
        self, batch: DocumentTensorDataModel
    ) -> ModelOutput:
        """Prepare the inputs for the explainer step."""
        logger.debug(
            "Preparing explanation inputs for sample_id: %s", batch.metadata.sample_id
        )

        # explained inputs
        inputs = self._perturbed_inputs(batch)

        # prepare additional forward args
        additional_forward_kwargs = (
            self._additional_forward_kwargs(batch) or OrderedDict()
        )

        # prepare baselines
        baselines = self._baselines(inputs)

        # prepare feature mask
        feature_mask, _ = self._feature_mask(
            inputs,
            word_ids=batch.word_ids,
            sequence_feature_keys=self._prepare_sequence_feature_keys(inputs),
        )

        # perturb the inputs
        perturbed_inputs = self._perturb_inputs(
            inputs=inputs, baselines=baselines, feature_masks=feature_mask
        )

        assert isinstance(self._model_pipeline, SequenceClassificationPipeline), (
            f"{self.__class__.__name__} can only be used with SequenceModelPipeline. Found {self._model_pipeline=}"
        )
        log_tensors_debug_info(
            perturbed_inputs, title="perturbedperturbed_inputs_batch"
        )
        log_tensors_debug_info(
            additional_forward_kwargs, title="additional_forward_kwargs"
        )
        model_output = self._model_pipeline._model(
            **perturbed_inputs, **additional_forward_kwargs
        )
        log_tensors_debug_info(model_output, title="model_output")
        loss = self._model_pipeline._get_loss(model_output=model_output)
        return self._output_transform(loss=loss, model_output=model_output, batch=batch)

    def _output_transform(
        self,
        loss: torch.Tensor,
        model_output: ModelOutput,
        batch: DocumentTensorDataModel,
    ) -> ModelOutput:
        return model_output


class SequenceClassificationFeaturePerturbationEvaluatorPipelineConfig(
    SequenceFeaturePerturbationEvaluatorPipelineConfig
):
    model_pipeline: SequenceClassificationPipelineConfig = (
        SequenceClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "sequence_classification"


@FEATURE_PERTURBATION_EVALUATOR_PIPELINES.register("sequence_classification")
class SequenceClassificationFeaturePerturbationEvaluatorPipeline(
    SequenceFeaturePerturbationEvaluatorPipeline
):
    __config__ = SequenceClassificationFeaturePerturbationEvaluatorPipelineConfig

    def _additional_forward_kwargs(self, batch):
        assert batch.labels is not None, (
            "Labels cannot be None for sequence classification"
        )
        additional_forward_kwargs = super()._additional_forward_kwargs(batch)
        return {**additional_forward_kwargs, "labels": batch.labels}

    def _output_transform(self, loss, model_output, batch):
        return self._model_pipeline._output_transform(
            loss=loss, model_output=model_output, batch=batch
        )


class TokenClassificationFeaturePerturbationEvaluatorPipelineConfig(
    SequenceFeaturePerturbationEvaluatorPipelineConfig
):
    model_pipeline: TokenClassificationPipelineConfig = (
        TokenClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "token_classification"


@FEATURE_PERTURBATION_EVALUATOR_PIPELINES.register("token_classification")
class TokenClassificationFeaturePerturbationEvaluatorPipeline(
    SequenceFeaturePerturbationEvaluatorPipeline
):
    __config__ = TokenClassificationFeaturePerturbationEvaluatorPipelineConfig

    def _additional_forward_kwargs(self, batch):
        assert batch.token_labels is not None, (
            "Labels cannot be None for token classification"
        )
        additional_forward_kwargs = super()._additional_forward_kwargs(batch)
        return {**additional_forward_kwargs, "labels": batch.token_labels}

    def _output_transform(self, loss, model_output, batch):
        return self._model_pipeline._output_transform(
            loss=loss, model_output=model_output, batch=batch
        )


class QuestionAnsweringFeaturePerturbationEvaluatorPipelineConfig(
    SequenceFeaturePerturbationEvaluatorPipelineConfig
):
    model_pipeline: QuestionAnsweringPipelineConfig = QuestionAnsweringPipelineConfig()

    @property
    def name(self) -> str:
        return "question_answering"


@FEATURE_PERTURBATION_EVALUATOR_PIPELINES.register("question_answering")
class QuestionAnsweringFeaturePerturbationEvaluatorPipeline(
    SequenceFeaturePerturbationEvaluatorPipeline
):
    __config__ = QuestionAnsweringFeaturePerturbationEvaluatorPipelineConfig

    def evaluation_step(
        self, perturbed_batch: dict[str, Any], additional_forward_kwargs: dict[str, Any]
    ) -> ModelOutput:
        return self._model_pipeline._model(perturbed_batch, **additional_forward_kwargs)
