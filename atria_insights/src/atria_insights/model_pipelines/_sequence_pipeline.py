from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Any, TypeVar

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._sequence_pipeline import (
    QuestionAnsweringPipelineConfig,
    SequenceClassificationPipelineConfig,
    SequenceModelPipeline,
    TokenClassificationPipelineConfig,
)
from atria_models.core.model_pipelines.utilities import log_tensor_info
from atria_models.core.models.transformers._models._encoder_model import (
    TransformersEncoderModel,
    TransformersEncoderModelOutput,
)
from atria_models.core.models.transformers._outputs import TokenClassificationHeadOutput
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_types._datasets import DatasetLabels
from pydantic import Field, model_validator
from torchxai.data_types import ExplanationTargetType, SingleTargetAcrossBatch

from atria_insights.baseline_generators._feature_based import (
    FeatureBasedBaselineGenerator,
    FeatureBasedBaselineGeneratorConfig,
)
from atria_insights.baseline_generators._sequence import SequenceBaselineGeneratorConfig
from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.data_types._targets import BatchExplanationTarget
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


class ExplainableSequenceModelPipelineConfig(ExplainableModelPipelineConfig):
    feature_segmentor: SequenceFeatureMaskSegmentorConfig = (
        SequenceFeatureMaskSegmentorConfig()
    )
    baseline_generator: (
        SequenceBaselineGeneratorConfig | FeatureBasedBaselineGeneratorConfig
    ) = SequenceBaselineGeneratorConfig()
    metric_baseline_generator: SequenceBaselineGeneratorConfig = (
        SequenceBaselineGeneratorConfig()
    )

    # only for occlusion explainer
    sliding_window_shapes_map: dict[str, tuple[int, ...]] | None = Field(
        default_factory=lambda: {
            "token_ids": (8,),
            "position_ids": (8,),
            "token_type_ids": (8,),
            "layout_ids": (8,),
            "image": (3, 16, 16),
        }
    )
    strides_map: dict[str, tuple[int, ...]] | None = Field(
        default_factory=lambda: {
            "token_ids": (4,),
            "position_ids": (4,),
            "token_type_ids": (4,),
            "layout_ids": (4,),
            "image": (3, 8, 8),
        }
    )
    ignored_feature_ids: list[str] = Field(default_factory=lambda: ["token_type_ids"])

    @model_validator(mode="after")
    def validate_configs(self) -> ExplainableSequenceModelPipelineConfig:
        if not isinstance(self.feature_segmentor, SequenceFeatureMaskSegmentorConfig):
            raise ValueError(
                "feature_segmentor must be an instance of SequenceFeatureMaskSegmentorConfig"
            )
        if not isinstance(
            self.baseline_generator,
            SequenceBaselineGeneratorConfig | FeatureBasedBaselineGeneratorConfig,
        ):
            raise ValueError(
                "baseline_generator must be an instance of SequenceBaselineGeneratorConfig or FeatureBasedBaselineGeneratorConfig"
            )

        return self


T_ExplainableSequenceModelPipelineConfig = TypeVar(
    "T_ExplainableSequenceModelPipelineConfig",
    bound="ExplainableSequenceModelPipelineConfig",
)


class ExplainableSequenceModelPipeline(
    ExplainableModelPipeline[
        ExplainableSequenceModelPipelineConfig, DocumentTensorDataModel
    ]
):
    __abstract__ = True
    __config__ = ExplainableSequenceModelPipelineConfig

    def __init__(
        self,
        config: ExplainableSequenceModelPipelineConfig,
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
        self._model_id_to_embeddings_inputs_list = []
        for param in inspect.signature(
            self._model_pipeline._model.ids_to_embeddings
        ).parameters.values():
            if param.name != "self":
                self._model_id_to_embeddings_inputs_list.append(param.name)

    def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
        return ExplainableSequenceModelForwardWrapper(model=model)

    def _validated_inputs(  # type: ignore[override]
        self,
        inputs: dict[str, torch.Tensor],
        additional_forward_kwargs: dict[str, Any] | None = None,
        baselines: dict[str, torch.Tensor] | None = None,
        metric_baselines: dict[str, torch.Tensor] | None = None,
        feature_mask: dict[str, torch.Tensor] | None = None,
        sliding_window_shapes: dict[str, tuple] | None = None,
        strides: dict[str, tuple] | None = None,
    ) -> tuple:
        """
        Validate and map inputs to the model forward signature.

        Returns:
            model_inputs: tuple of positional arguments for model forward
            expected_params: list of expected parameter names (excluding self)
        """
        additional_forward_kwargs = additional_forward_kwargs or {}

        # ---- inputs ----
        baselines_tuple = None
        feature_mask_tuple = None
        metric_baselines_tuple = None
        feature_keys = tuple(inputs.keys())
        feature_values = tuple(inputs.values())
        if baselines is not None:
            assert isinstance(baselines, dict), (
                "If inputs is an dict, baselines must also be an dict."
            )

            baselines_tuple = ()
            for input_key, input_value in inputs.items():
                baseline = baselines[input_key]

                # assert shape matches
                # Note: baseline batch size can be different due to multiple baselines
                assert baseline.shape[1:] == input_value.shape[1:], (
                    f"Baseline shape {baseline.shape} does not match input shape {input_value.shape} for key {input_key}"
                )

                baselines_tuple += (baseline,)  # type: ignore

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

        sliding_window_shapes_tuple = None
        if sliding_window_shapes is not None:
            sliding_window_shapes = {
                key: sliding_window_shapes[key] for key in feature_keys
            }

            # make sure the shape matches the input shape
            sliding_window_shapes_tuple = ()
            for input_key, input_value in inputs.items():
                # we take the shape of the a single sample
                single_input_shape = input_value.shape[1:]

                # expand the shape of the sliding window to match the input shape
                sliding_window_shape = sliding_window_shapes[input_key]
                for dim in single_input_shape[len(sliding_window_shape) :]:
                    sliding_window_shape += (dim,)  # type: ignore
                sliding_window_shapes_tuple += (sliding_window_shape,)  # type: ignore

        strides_tuple = None
        if strides is not None:
            strides = {key: strides[key] for key in feature_keys}
            strides_tuple = ()
            for input_key, input_value in inputs.items():
                # we take the shape of the a single sample
                single_input_shape = input_value.shape[1:]

                # expand the shape of the sliding window to match the input shape
                strides_shape = strides[input_key]
                for dim in single_input_shape[len(strides_shape) :]:
                    strides_shape += (dim,)
                strides_tuple += (strides_shape,)  # type: ignore

        # finally we remap the feature keys from ids to embeddings
        feature_keys = tuple(key.replace("_ids", "_embeddings") for key in feature_keys)
        if "token_type_ids" in additional_forward_kwargs:
            additional_forward_kwargs["token_type_embeddings"] = (
                additional_forward_kwargs.pop("token_type_ids")
            )

        args_mapping = list(feature_keys) + list(additional_forward_kwargs.keys())
        additional_forward_args = tuple(additional_forward_kwargs.values()) + (
            args_mapping,
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
            baselines_tuple,
            metric_baselines_tuple,
            feature_mask_tuple,
            sliding_window_shapes_tuple,
            strides_tuple,
            feature_keys,
            args_mapping,
        )

    def _build_feature_segmentor(self):
        assert isinstance(self._model_pipeline._model, TransformersEncoderModel)
        self._feature_segmentor = self.config.feature_segmentor.build(
            special_token_ids=self._model_pipeline._model.config.embeddings_config.special_token_ids
        )

    def _build_baseline_generator(self):
        # build baselines generator
        if isinstance(self.config.baseline_generator, SequenceBaselineGeneratorConfig):
            self._baseline_generator = self.config.baseline_generator.build(
                model=self._model_pipeline._model
            )
        else:
            self._baseline_generator = self.config.baseline_generator.build()

        # build metric baselines generator
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

    def _explained_inputs(  # type: ignore[override]
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

    def _target(  # type: ignore[override]
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

    def _baselines(  # type: ignore[override]
        self, explained_inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Generate baselines for the explainer."""
        logger.debug(
            "Generating baselines using baseline generator with config: %s",
            self.config.baseline_generator,
        )
        baselines = self._baseline_generator(explained_inputs, **kwargs)

        # if the baseline generator is feature based
        if isinstance(self._baseline_generator, FeatureBasedBaselineGenerator):
            # make sure we only return baselines for the input keys
            sequence_baselines = self._model_pipeline._model.ids_to_embeddings(
                **{k: v for k, v in baselines.items() if k != "image"}
            ).to_id_map()
            for key in baselines.keys():
                if key in sequence_baselines:
                    baselines[key] = sequence_baselines[key]

        # filter out ignored feature ids from baselines
        baselines = {
            k: v
            for k, v in baselines.items()
            if k not in self.config.ignored_feature_ids
        }

        return baselines

    def _metric_baselines(self, explained_inputs: dict[str, torch.Tensor], **kwargs):
        """Generate baselines for the explainer."""
        logger.debug(
            "Generating baselines using baseline generator with config: %s",
            self.config.metric_baseline_generator,
        )
        baselines = self._metric_baselines_generator(explained_inputs, **kwargs)

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

    def _sliding_window_shapes_and_strides(
        self, input_feature_keys: list[str]
    ) -> tuple[dict[str, tuple] | None, dict[str, tuple] | None]:
        if "sliding_window_shapes" not in self._explainer_args:
            return None, None
        if (
            self.config.sliding_window_shapes_map is None
            or self.config.strides_map is None
        ):
            raise ValueError(
                f"sliding_window_shapes_map and strides_map must be defined in the config for {self._explainer.__class__.__name__}."
            )
        sliding_window_shapes_map = {}
        strides = {}
        for key in input_feature_keys:
            if key in self.config.ignored_feature_ids:
                continue
            sliding_window_shapes_map[key] = self.config.sliding_window_shapes_map[key]
            strides[key] = self.config.strides_map[key]

        return sliding_window_shapes_map, strides

    def _prepare_sequence_feature_keys(
        self, explained_inputs: dict[str, torch.Tensor]
    ) -> list[str]:
        possible_feature_keys = []
        for key in ["token_ids", "position_ids", "layout_ids", "token_type_ids"]:
            if key in explained_inputs and key not in self.config.ignored_feature_ids:
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

            # prepare baselines
            baselines = self._baselines(inputs)

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

            # prepare sliding window shapes map and strides map for occlusion explainer
            sliding_window_shapes, strides = self._sliding_window_shapes_and_strides(
                input_feature_keys=list(inputs.keys())
            )

            # map inputs to embeddings
            input_embeddings = self._model_pipeline._model.ids_to_embeddings(
                **{key: inputs[key] for key in self._model_id_to_embeddings_inputs_list}
            ).to_id_map()

            # filter out ignored feature ids from input embeddings and add them to additional forward kwargs
            for key in self.config.ignored_feature_ids:
                embeddings = input_embeddings.pop(key)
                additional_forward_kwargs = {
                    key: embeddings,
                    **additional_forward_kwargs,
                }

            # nowe remake inputs by replacing ids with embeddings
            finalized_inputs = {}
            for key in inputs.keys():
                if key in self.config.ignored_feature_ids:
                    continue
                if key in input_embeddings:
                    finalized_inputs[key] = input_embeddings[key]
                else:
                    finalized_inputs[key] = inputs[key]
            inputs = finalized_inputs

            # now log info
            log_tensor_info(inputs, name="inputs")
            log_tensor_info(additional_forward_kwargs, name="additional_forward_kwargs")
            log_tensor_info(baselines, name="baselines")
            if metric_baselines is not None:
                log_tensor_info(metric_baselines, name="metric_baselines")
            log_tensor_info(feature_mask, name="feature_mask")
            log_tensor_info(sliding_window_shapes, name="sliding_window_shapes")
            log_tensor_info(strides, name="strides")

            (
                inputs_tuple,
                additional_forward_args,
                baselines_tuple,
                metric_baselines_tuple,
                feature_mask_tuple,
                sliding_window_shapes_tuple,
                strides_tuple,
                feature_keys,
                _,
            ) = self._validated_inputs(
                inputs=inputs,
                additional_forward_kwargs=additional_forward_kwargs,
                baselines=baselines,
                metric_baselines=metric_baselines,
                feature_mask=feature_mask,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
            )

            # forward pass
            model_outputs = self._wrapped_model(
                *(*inputs_tuple, *additional_forward_args)
            )

            # prepare target
            target = self._target(batch=batch, model_outputs=model_outputs)

            # prepare explanation inputs
            return model_outputs, BatchExplanationInputs(
                sample_id=batch.metadata.sample_id,
                inputs=inputs_tuple,
                additional_forward_args=additional_forward_args,
                baselines=baselines_tuple
                if "baselines" in self._explainer_args
                else None,
                metric_baselines=metric_baselines_tuple,
                feature_mask=feature_mask_tuple
                if "feature_mask" in self._explainer_args
                else None,
                metric_feature_mask=feature_mask_tuple,
                target=target,
                sliding_window_shapes=sliding_window_shapes_tuple,
                strides=strides_tuple,
                frozen_features=frozen_features,
                feature_keys=feature_keys,
            )


class ExplainableSequenceClassificationPipelineConfig(
    ExplainableSequenceModelPipelineConfig
):
    model_pipeline: SequenceClassificationPipelineConfig = (
        SequenceClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "sequence_classification"


@EXPLAINABLE_MODEL_PIPELINES.register("sequence_classification")
class ExplainableSequenceClassificationPipeline(ExplainableSequenceModelPipeline):
    __config__ = ExplainableSequenceClassificationPipelineConfig


class ExplainableTokenClassificationPipelineConfig(
    ExplainableSequenceModelPipelineConfig
):
    model_pipeline: TokenClassificationPipelineConfig = (
        TokenClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "token_classification"


@EXPLAINABLE_MODEL_PIPELINES.register("token_classification")
class ExplainableTokenClassificationPipeline(ExplainableSequenceModelPipeline):
    __config__ = ExplainableTokenClassificationPipelineConfig

    def _target(
        self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
    ) -> ExplanationTargetType | list[ExplanationTargetType]:
        if (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.ground_truth
        ):
            raise ValueError(
                "Ground truth explanation targets are not supported for token classification tasks."
            )
        elif (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.predicted
        ):
            # model_outputs is returned predictions of shape [batch_size, seq_len] tensor so a target for each token
            return [
                SingleTargetAcrossBatch(index=i) for i in range(model_outputs.shape[1])
            ]
        else:
            raise ValueError(
                "Only 'predicted' explanation target strategy is supported for token classification tasks."
            )

    def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
        class WrappedModel(torch.nn.Module):
            def __init__(self, model: torch.nn.Module):
                super().__init__()
                self.model = model
                self.model_signature = inspect.signature(model.forward)

            def forward(self, *args) -> torch.Tensor:
                args_mapping = args[-1]  # the last arg is the args mapping
                assert isinstance(args_mapping, list), (
                    f"Expected args_mapping to be a list of keys, got {type(args_mapping)}"
                )
                args = args[:-1]  # all but last are the actual model args
                logger.debug(
                    f"WrappedModel.forward called with {len(args)} args and args_mapping: {args_mapping}"
                )

                if len(args) != len(args_mapping):
                    raise ValueError(
                        f"Expected {len(args_mapping)} inputs, but got {len(args)} inputs."
                    )

                model_kwargs = {
                    args_mapping[i]: args[i] for i in range(len(args_mapping))
                }

                # we remap embedding args to id_or_embedding args
                # inside the model, we need to remap ids -> embeddings
                for key in [
                    "token_ids",
                    "position_ids",
                    "layout_ids",
                    "token_type_ids",
                ]:
                    if key in model_kwargs:
                        model_kwargs[key.replace("_ids", "_ids_or_embeddings")] = (
                            model_kwargs.pop(key)
                        )

                # filter unsupported args (important for heterogeneous models)
                model_kwargs = {
                    k: v
                    for k, v in model_kwargs.items()
                    if k in self.model_signature.parameters
                }
                outputs = self.model(**model_kwargs, is_embedding=True)

                assert isinstance(outputs, TransformersEncoderModelOutput)
                assert isinstance(outputs.head_output, TokenClassificationHeadOutput)
                assert outputs.head_output.logits is not None
                probs = self.softmax(outputs.head_output.logits)
                probs = torch.gather(
                    probs, 2, probs.argmax(dim=-1).unsqueeze(-1)
                ).squeeze(-1)
                return probs

        return WrappedModel(model=model)


class ExplainableQuestionAnsweringPipelineConfig(
    ExplainableSequenceModelPipelineConfig
):
    model_pipeline: QuestionAnsweringPipelineConfig = QuestionAnsweringPipelineConfig()

    @property
    def name(self) -> str:
        return "question_answering"


@EXPLAINABLE_MODEL_PIPELINES.register("question_answering")
class ExplainableQuestionAnsweringPipeline(ExplainableSequenceModelPipeline):
    __config__ = ExplainableQuestionAnsweringPipelineConfig

    def _target(
        self, batch: DocumentTensorDataModel, model_outputs: torch.Tensor
    ) -> ExplanationTargetType | list[ExplanationTargetType]:
        if (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.ground_truth
        ):
            raise ValueError(
                "Ground truth explanation targets are not supported for token classification tasks."
            )
        elif (
            self.config.explanation_target_strategy
            == ExplanationTargetStrategy.predicted
        ):
            # model_outputs is returned predictions of shape [batch_size, seq_len] tensor so a target for each token
            return [
                SingleTargetAcrossBatch(index=i) for i in range(1)
            ]  # for q/a we explain the start token only
        else:
            raise ValueError(
                f"Explanation target strategy {self.config.explanation_target_strategy} is not supported for question answering tasks."
            )

    def _model_forward(
        self,
        explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
        additional_forward_args: dict[str, Any] | None,
    ) -> torch.Tensor:
        from torch.nn.functional import softmax

        model_outputs = self._model_pipeline._model(
            explained_inputs, **(additional_forward_args or {})
        )
        if isinstance(model_outputs, dict):
            logits = model_outputs["logits"]
        elif hasattr(model_outputs, "logits"):
            logits = model_outputs.logits
        else:
            logits = model_outputs
        return softmax(logits, dim=-1)

    def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
        class WrappedModel(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self._model = model

            def forward(
                self,
                explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
                additional_forward_args: dict[str, Any] | None = None,
            ) -> torch.Tensor:
                from torch.nn.functional import softmax

                model_outputs = self._model(
                    explained_inputs, **(additional_forward_args or {})
                )
                start_pred = softmax(model_outputs.start_logits, dim=-1)
                end_pred = softmax(model_outputs.end_logits, dim=-1)
                return torch.cat([start_pred, end_pred])

        return WrappedModel(model)
