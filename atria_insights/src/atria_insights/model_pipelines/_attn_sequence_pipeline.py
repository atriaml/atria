from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Any, TypeVar

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._sequence_pipeline import (
    SequenceClassificationPipelineConfig,
    SequenceModelPipeline,
)
from atria_models.core.model_pipelines.utilities import log_tensor_info
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
)
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_types._datasets import DatasetLabels
from pydantic import model_validator

from atria_insights.baseline_generators._sequence import (
    NoEmbedSequenceBaselineGeneratorConfig,
)
from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.data_types._targets import BatchExplanationTarget
from atria_insights.explainers._attn._config import (
    AttnExplainerConfigType,
    RawAttentionExplainerConfig,
)
from atria_insights.explainers._attn._target import AttentionTokenTarget
from atria_insights.feature_segmentors._sequence import (
    SequenceFeatureMaskSegmentorConfig,
)
from atria_insights.model_pipelines._common import (
    ExplainableModelPipelineConfig,
    ExplanationTargetStrategy,
)
from atria_insights.model_pipelines._forward_wrappers._sequence_forward_wrappers import (
    ExplainableSequenceModelForwardWrapper,
)
from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline
from atria_insights.model_pipelines._registry_groups import EXPLAINABLE_MODEL_PIPELINES

logger = get_logger(__name__)


class AttnExplainableSequenceModelPipelineConfig(ExplainableModelPipelineConfig):
    explainer: AttnExplainerConfigType = RawAttentionExplainerConfig()
    feature_segmentor: SequenceFeatureMaskSegmentorConfig = (
        SequenceFeatureMaskSegmentorConfig()
    )
    metric_baseline_generator: NoEmbedSequenceBaselineGeneratorConfig = (
        NoEmbedSequenceBaselineGeneratorConfig()
    )

    @model_validator(mode="after")
    def validate_configs(self) -> AttnExplainableSequenceModelPipelineConfig:
        if not isinstance(self.feature_segmentor, SequenceFeatureMaskSegmentorConfig):
            raise ValueError(
                "feature_segmentor must be an instance of SequenceFeatureMaskSegmentorConfig"
            )
        if not isinstance(
            self.metric_baseline_generator, NoEmbedSequenceBaselineGeneratorConfig
        ):
            raise ValueError(
                "metric_baseline_generator must be an instance of NoEmbedSequenceBaselineGeneratorConfig"
            )

        return self


T_AttnExplainableSequenceModelPipelineConfig = TypeVar(
    "T_AttnExplainableSequenceModelPipelineConfig",
    bound="AttnExplainableSequenceModelPipelineConfig",
)


class AttnExplainableSequenceModelPipeline(
    ExplainableModelPipeline[
        T_AttnExplainableSequenceModelPipelineConfig, DocumentTensorDataModel
    ]
):
    __abstract__ = True

    def __init__(
        self,
        config: AttnExplainableSequenceModelPipelineConfig,
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

    def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
        return ExplainableSequenceModelForwardWrapper(model=model, is_embedding=False)

    def _validated_inputs(  # type: ignore[override]
        self,
        inputs: dict[str, torch.Tensor],
        additional_forward_kwargs: dict[str, Any] | None = None,
        metric_baselines: dict[str, torch.Tensor] | None = None,
        feature_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple:
        """
        Validate and map inputs to the model forward signature.

        Returns:
            model_inputs: tuple of positional arguments for model forward
            expected_params: list of expected parameter names (excluding self)
        """
        additional_forward_kwargs = additional_forward_kwargs or {}

        # ---- inputs ----
        feature_mask_tuple = None
        metric_baselines_tuple = None
        feature_keys = tuple(inputs.keys())
        feature_values = tuple(inputs.values())
        if metric_baselines is not None:
            assert isinstance(metric_baselines, dict), (
                "If inputs is an dict, metric_baselines must also be an dict."
            )
            metric_baselines_tuple = ()
            for input_key, input_value in inputs.items():
                baseline = metric_baselines[input_key]

                # assert shape matches
                # Note: baseline batch size can be different due to multiple baselines
                assert baseline.shape[1:] == input_value.shape[1:], (
                    f"Metric baseline shape {baseline.shape} does not match input shape {input_value.shape} for key {input_key}"
                )

                metric_baselines_tuple += (baseline,)  # type: ignore

        if feature_mask is not None:
            assert isinstance(feature_mask, dict), (
                "If inputs is an dict, feature_mask must also be an dict."
            )

            # expand feature masks to match input shapes
            feature_mask_tuple = ()
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

                feature_mask_tuple += (mask,)  # type: ignore

        args_mapping = list(feature_keys) + list(additional_forward_kwargs.keys())
        bsz = feature_values[0].shape[0]
        additional_forward_args = tuple(additional_forward_kwargs.values()) + (
            [args_mapping for _ in range(bsz)],
        )
        assert len(inputs) == len(inputs.keys()), (
            "Input feature keys length does not match inputs length."
            f" {len(inputs.keys())=}, {len(inputs)=}"
        )
        assert len(additional_forward_args) + len(inputs) == len(args_mapping) + 1, (
            "Args map length does not match inputs and additional forward args length."
            f" {len(args_mapping)=}, {len(inputs)=}, {len(additional_forward_args)=}"
        )
        return (
            feature_values,
            additional_forward_args,
            metric_baselines_tuple,
            feature_mask_tuple,
            feature_keys,
            args_mapping,
        )

    def _build_explainer(self):
        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)
        # build model with wrapped forward
        self._model_signature = inspect.signature(self._model_pipeline._model.forward)
        self._wrapped_model = self._wrap_model_forward(self._model_pipeline._model)

        # build explainer
        self._explainer = self.config.explainer.build(model=self._wrapped_model)

        # get possible explainer args
        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        self._explainer_args = inspect.signature(
            self._explainer.explain
        ).parameters.keys()

        # for attention explainers we need to get the feature ids from the model itself
        self._attn_feature_ids = self._model_pipeline._model.attn_feature_ids()

    def _build_feature_segmentor(self):
        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)
        self._feature_segmentor = self.config.feature_segmentor.build(
            special_token_ids=self._model_pipeline._model.config.embeddings_config.special_token_ids
        )

    def _build_baseline_generator(self):
        self._metric_baselines_generator = self.config.metric_baseline_generator.build(
            model=self._model_pipeline._model
        )

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
        ):
            assert batch.image is not None, "Image cannot be None"
            inputs["image"] = batch.image

        if self._model_pipeline.config.use_bbox and (
            "layout_ids" in self._model_signature.parameters.keys()
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

    def _explained_inputs(  # type: ignore[override]
        self, batch: DocumentTensorDataModel
    ) -> dict[str, torch.Tensor]:
        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)

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
        explained_inputs = {
            "token_ids": batch.token_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

        if self._model_pipeline.config.use_image and (
            "image" in self._model_signature.parameters.keys()
        ):
            assert batch.image is not None, "Image cannot be None"
            explained_inputs["image"] = batch.image

        if self._model_pipeline.config.use_bbox and (
            "layout_ids_or_embeddings" in self._model_signature.parameters.keys()
        ):
            assert batch.token_bboxes is not None, "Token bboxes cannot be None"
            token_bboxes = batch.token_bboxes
            if batch.metadata.bbox_normalized[0] and token_bboxes is not None:
                token_bboxes = (
                    (token_bboxes * 1000.0).clip(0, 1000).long()
                    if token_bboxes is not None
                    else None
                )

            explained_inputs["layout_ids"] = token_bboxes
        return explained_inputs

    def _additional_forward_kwargs(self, batch: DocumentTensorDataModel):  # type: ignore[override]
        additional_forward_kwargs = {"attention_mask": batch.attention_mask}
        return additional_forward_kwargs

    def _target(
        self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
    ) -> BatchExplanationTarget | list[BatchExplanationTarget]:
        assert self._model_pipeline._labels.classification is not None, (
            "Labels are required for explanation target strategies other than 'predicted'."
        )
        label_names = self._model_pipeline._labels.classification
        if (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.ground_truth
        ):
            assert batch.label is not None, (
                "Ground truth labels are required for explanation target strategies other than 'predicted'."
            )
            return BatchExplanationTarget(
                value=batch.label.tolist(),
                name=[label_names[idx] for idx in batch.label.tolist()],
            )
        elif (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.predicted
        ):
            predictions = model_outputs.argmax(dim=-1)
            prediction_label_names = [label_names[idx] for idx in predictions]
            return BatchExplanationTarget(
                value=predictions.tolist(), name=prediction_label_names
            )
        else:
            # in case of 'all' we compute the explanations for all classes
            total_labels = model_outputs.shape[1]
            batch_size = model_outputs.shape[0]
            return [
                BatchExplanationTarget(
                    value=[label_index for _ in range(batch_size)],
                    name=[label_names[label_index] for _ in range(batch_size)],
                )
                for label_index in range(total_labels)
            ]

    def _attn_token_target(
        self, batch: DocumentTensorDataModel
    ) -> AttentionTokenTarget:
        # defaults to CLS token explanation target
        batch_size = batch.token_ids.shape[0]
        return AttentionTokenTarget(indices=[[0] for _ in range(batch_size)])

    def _metric_baselines(self, explained_inputs: dict[str, torch.Tensor], **kwargs):
        """Generate baselines for the explainer."""
        logger.debug(
            "Generating baselines using baseline generator with config: %s",
            self.config.metric_baseline_generator,
        )
        baselines = self._metric_baselines_generator(explained_inputs, **kwargs)

        # filter out ignored feature ids from baselines
        baselines = {k: v for k, v in baselines.items() if k in self._attn_feature_ids}

        return baselines

    def _feature_mask(  # type: ignore[override]
        self,
        explained_inputs: dict[str, torch.Tensor],
        word_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
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
            sequence_ids=sequence_ids,
            sequence_feature_keys=sequence_feature_keys,
        )
        return feature_masks, frozen_features

    def _prepare_sequence_feature_keys(
        self, explained_inputs: dict[str, torch.Tensor]
    ) -> list[str]:
        possible_feature_keys = []
        for key in ["token_ids", "position_ids", "layout_ids", "token_type_ids"]:
            if key in explained_inputs and key in self._attn_feature_ids:
                possible_feature_keys.append(key)
        return possible_feature_keys

    def prepare_explanation_inputs(
        self, batch: DocumentTensorDataModel
    ) -> tuple[Any, BatchExplanationInputs]:
        """Prepare the inputs for the explainer step."""
        with torch.no_grad():
            # prepare explained inputs
            # we need input ids here for baseline generation
            logger.debug(
                "Preparing explanation inputs for sample_id: %s",
                batch.metadata.sample_id,
            )

            # explained inputs
            inputs = self._explained_inputs(batch)

            # prepare additional forward args
            additional_forward_kwargs = (
                self._additional_forward_kwargs(batch) or OrderedDict()
            )

            # prepare baselines for metrics if needed
            metric_baselines = None
            if self.config.explainability_metrics is not None:
                metric_baselines = self._metric_baselines(inputs)

            # prepare feature mask
            feature_mask, frozen_features = self._feature_mask(
                inputs,
                word_ids=batch.word_ids,
                sequence_ids=batch.sequence_ids,
                sequence_feature_keys=self._prepare_sequence_feature_keys(inputs),
            )

            # filter out ignored feature ids from input embeddings and add them to additional forward kwargs
            for key in list(inputs.keys()):
                if key in self._attn_feature_ids:
                    continue
                ignored_input = inputs.pop(key)
                additional_forward_kwargs = {
                    key: ignored_input,
                    **additional_forward_kwargs,
                }

            # now log info
            log_tensor_info(inputs, name="inputs")
            log_tensor_info(additional_forward_kwargs, name="additional_forward_kwargs")
            if metric_baselines is not None:
                log_tensor_info(metric_baselines, name="metric_baselines")
            log_tensor_info(feature_mask, name="feature_mask")

            (
                inputs_tuple,
                additional_forward_args,
                metric_baselines_tuple,
                feature_mask_tuple,
                feature_keys,
                _,
            ) = self._validated_inputs(
                inputs=inputs,
                additional_forward_kwargs=additional_forward_kwargs,
                metric_baselines=metric_baselines,
                feature_mask=feature_mask,
            )

            # forward pass
            model_outputs = self._wrapped_model(
                *(*inputs_tuple, *additional_forward_args)
            )

            # prepare target this target is used to get probabilties, and is only used for downstream metric evals
            # for attention, we need to separately prepare the attention target which is used to select the attention scores for the target tokens
            target = self._target(batch=batch, model_outputs=model_outputs)

            # for attention explainers, we default to explaining the attention to the CLS token, so we prepare a separate target for that as well
            attention_token_target = self._attn_token_target(batch=batch)

            # prepare explanation inputs
            return model_outputs, BatchExplanationInputs(
                sample_id=batch.metadata.sample_id,
                inputs=inputs_tuple,
                additional_forward_args=additional_forward_args,
                metric_baselines=metric_baselines_tuple,
                feature_mask=feature_mask_tuple
                if "feature_mask" in self._explainer_args
                else None,
                metric_feature_mask=feature_mask_tuple,
                target=target,
                attention_token_target=attention_token_target,
                frozen_features=frozen_features,
                feature_keys=feature_keys,
            )

    def explainer_forward(
        self, explanation_inputs: BatchExplanationInputs
    ) -> tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]]:
        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        kwargs = {}
        for arg in self._explainer_args:
            kwargs[arg] = getattr(explanation_inputs, arg)

        logger.debug(f"Running explainer {self._explainer} forward with inputs:")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                logger.debug(
                    f"  {k}: Tensor shape {v.shape}, dtype {v.dtype}, device {v.device}"
                )
            elif isinstance(v, tuple):
                for idx, item in enumerate(v):
                    if isinstance(item, torch.Tensor):
                        logger.debug(
                            f"  {k}.{idx}: Tensor shape {item.shape}, dtype {item.dtype}, device {item.device}"
                        )
                    else:
                        logger.debug(f"  {k}.{idx}: {type(item)}")
            else:
                logger.debug(f"  {k}: {type(v)}")

        self._explainer._model.return_attns = True
        explanations = self._explainer.explain(**kwargs)
        self._explainer._model.return_attns = False

        # validated explanations
        validated_explanations = []
        if isinstance(explanations, tuple):
            return explanations
        elif isinstance(explanations, list):
            for exp in explanations:
                if not isinstance(exp, tuple):
                    raise ValueError(
                        "Explainer returned a list but elements are not tuples."
                    )
                validated_explanations.append(exp)
            return validated_explanations
        else:
            raise ValueError(
                "Explainer returned invalid type. Expected tuple or list of tuples."
            )


class AttnExplainableSequenceClassificationPipelineConfig(
    AttnExplainableSequenceModelPipelineConfig
):
    model_pipeline: SequenceClassificationPipelineConfig = (
        SequenceClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "sequence_classification"


@EXPLAINABLE_MODEL_PIPELINES.register("attn_sequence_classification")
class ExplainableSequenceClassificationPipeline(
    AttnExplainableSequenceModelPipeline[
        AttnExplainableSequenceClassificationPipelineConfig
    ]
):
    __config__ = AttnExplainableSequenceClassificationPipelineConfig


# class ExplainableTokenClassificationPipelineConfig(
#     ExplainableSequenceModelPipelineConfig
# ):
#     __hash_exclude__: ClassVar[set[str]] = {
#         "explainability_metrics",
#         "iterative_computation",
#         "internal_batch_size",
#         "grad_batch_size",
#         "throw_on_load_mismatch",
#         "remove_other_labels",
#         "profile_time",
#     }

#     model_pipeline: TokenClassificationPipelineConfig = (
#         TokenClassificationPipelineConfig()
#     )
#     use_word_level_targets: bool = True
#     remove_other_labels: bool = False

#     @property
#     def name(self) -> str:
#         return "token_classification"


# @EXPLAINABLE_MODEL_PIPELINES.register("token_classification")
# class ExplainableTokenClassificationPipeline(
#     ExplainableSequenceModelPipeline[ExplainableTokenClassificationPipelineConfig]
# ):
#     __config__ = ExplainableTokenClassificationPipelineConfig

#     def _target(
#         self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
#     ) -> BatchExplanationTarget | list[BatchExplanationTarget]:
#         if self.config.explanation_target_strategy in [
#             ExplanationTargetStrategy.ground_truth,
#             ExplanationTargetStrategy.all,
#         ]:
#             # for token level tasks we do not support ground truth explanation targets
#             # as the forward wrapper returns per token predicted logits
#             raise ValueError(
#                 "'ground_truth' and 'all' explanation target strategies are not supported for token classification tasks."
#             )

#         # the token classification forward wrapper always returns the per token predicted label logits
#         # so model_outputs is of shape [batch_size, seq_len] => a logit for each token
#         if self.config.use_word_level_targets:
#             # for word level targets per word instead of generating targets for each token,
#             # we get the word ids and generate targets per word since models are usually trained with only
#             # first token of each word having a label
#             batch_size = model_outputs.shape[0]
#             assert batch_size == 1, (
#                 f"Word level targets are only supported for batch size of 1. Found {batch_size=}"
#                 f"This is because word ids are different for each sample in the batch and results in varying target shapes "
#                 f"for each sample in the batch. Since for multiple targets, we use multi-target mode all samples"
#                 f"must have equal number of targets which is not possible with per-target-mode unless some sort of padding "
#                 f"is introduced."
#             )
#             sample_word_ids = batch.word_ids[0]
#             token_labels = batch.token_labels[0]
#             target = [
#                 BatchExplanationTarget(value=[index], name=[str(index)])
#                 for index in _generate_word_level_targets(
#                     word_ids_per_sample=sample_word_ids,
#                     token_labels_per_sample=token_labels,
#                     remove_other_labels=self.config.remove_other_labels,
#                 )
#             ]
#             return target
#         else:
#             # otherwise we create explanation targets for each token
#             return [
#                 BatchExplanationTarget(
#                     value=[i for _ in range(model_outputs.shape[0])],
#                     name=[str(i) for _ in range(model_outputs.shape[0])],
#                 )
#                 for i in range(model_outputs.shape[1])
#             ]

#     def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
#         return ExplainableTokenClassificationModelForwardWrapper(model=model)


# class ExplainableLayoutTokenClassificationPipelineConfig(
#     ExplainableSequenceModelPipelineConfig
# ):
#     model_pipeline: LayoutTokenClassificationPipelineConfig = (
#         LayoutTokenClassificationPipelineConfig()
#     )
#     use_word_level_targets: bool = True

#     @property
#     def name(self) -> str:
#         return "layout_token_classification"


# @EXPLAINABLE_MODEL_PIPELINES.register("layout_token_classification")
# class ExplainableLayoutTokenClassificationPipeline(
#     ExplainableSequenceModelPipeline[ExplainableLayoutTokenClassificationPipelineConfig]
# ):
#     __config__ = ExplainableLayoutTokenClassificationPipelineConfig

#     def _target(
#         self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
#     ) -> BatchExplanationTarget | list[BatchExplanationTarget]:
#         if self.config.explanation_target_strategy in [
#             ExplanationTargetStrategy.ground_truth,
#             ExplanationTargetStrategy.all,
#         ]:
#             # for token level tasks we do not support ground truth explanation targets
#             # as the forward wrapper returns per token predicted logits
#             raise ValueError(
#                 "'ground_truth' and 'all' explanation target strategies are not supported for token classification tasks."
#             )

#         # the token classification forward wrapper always returns the per token predicted label logits
#         # so model_outputs is of shape [batch_size, seq_len] => a logit for each token
#         if self.config.use_word_level_targets:
#             # for word level targets per word instead of generating targets for each token,
#             # we get the word ids and generate targets per word since models are usually trained with only
#             # first token of each word having a label
#             batch_size = model_outputs.shape[0]
#             assert batch_size == 1, (
#                 f"Word level targets are only supported for batch size of 1. Found {batch_size=}"
#                 f"This is because word ids are different for each sample in the batch and results in varying target shapes "
#                 f"for each sample in the batch. Since for multiple targets, we use multi-target mode all samples"
#                 f"must have equal number of targets which is not possible with per-target-mode unless some sort of padding "
#                 f"is introduced."
#             )
#             sample_word_ids = batch.word_ids[0]
#             return [
#                 BatchExplanationTarget(value=[index], name=[str(index)])
#                 for index in _generate_word_level_targets(sample_word_ids)
#             ]
#         else:
#             # otherwise we create explanation targets for each token
#             return [
#                 BatchExplanationTarget(
#                     value=[i for _ in range(model_outputs.shape[0])],
#                     name=[str(i) for _ in range(model_outputs.shape[0])],
#                 )
#                 for i in range(model_outputs.shape[1])
#             ]

#     def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
#         return ExplainableTokenClassificationModelForwardWrapper(model=model)


# class ExplainableQuestionAnsweringPipelineConfig(
#     ExplainableSequenceModelPipelineConfig
# ):
#     model_pipeline: QuestionAnsweringPipelineConfig = QuestionAnsweringPipelineConfig()

#     @property
#     def name(self) -> str:
#         return "question_answering"


# @EXPLAINABLE_MODEL_PIPELINES.register("question_answering")
# class ExplainableQuestionAnsweringPipeline(ExplainableSequenceModelPipeline):
#     __config__ = ExplainableQuestionAnsweringPipelineConfig

#     def _target(
#         self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
#     ) -> BatchExplanationTarget | list[BatchExplanationTarget]:
#         if self.config.explanation_target_strategy in [
#             ExplanationTargetStrategy.ground_truth,
#             ExplanationTargetStrategy.all,
#         ]:
#             # for token level tasks we do not support ground truth explanation targets
#             # as the forward wrapper returns per token predicted logits
#             raise ValueError(
#                 "'ground_truth' and 'all' explanation target strategies are not supported for token classification tasks."
#             )

#         # for question answering forward wrapper, model_outputs is of shape [batch_size, 2]
#         # where the last dimension contains start and end token probabilities
#         # therefore the first target for each sample is the logits of the start token and
#         # the second is the logits of the end token
#         batch_size = model_outputs.shape[0]
#         return [
#             BatchExplanationTarget(
#                 value=[0 for _ in range(batch_size)],
#                 name=["start" for _ in range(batch_size)],
#             ),
#             BatchExplanationTarget(
#                 value=[1 for _ in range(batch_size)],
#                 name=["end" for _ in range(batch_size)],
#             ),
#         ]

#     def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
#         return ExplainableQuestionAnsweringModelForwardWrapper(model=model)
