from __future__ import annotations

import inspect
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generic

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._ops import ModelPipelineOps
from atria_models.core.model_pipelines.utilities import log_tensor_info
from atria_registry._module_base import ConfigurableModule
from atria_transforms.core._data_types._base import T_TensorDataModel
from atria_types._datasets import DatasetLabels
from ignite.metrics import Metric
from torchxai.data_types._target import SingleTargetPerSample
from tqdm import tqdm

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.data_types._explanation_state import (
    BatchExplanation,
    BatchExplanationState,
    ComputeMetrics,
    MultiTargetBatchExplanation,
)
from atria_insights.data_types._targets import BatchExplanationTarget
from atria_insights.engines._explanation_step import ExplanationStepOutput
from atria_insights.explanation_pipelines._common import T_ExplanationPipelineConfig
from atria_insights.storage.sample_cache_managers._explanation_state import (
    ExplanationStateCacher,
)

logger = get_logger(__name__)

_DEFAULT_FEATURE_INPUT_KEY = "input_feature"


class ExplanationPipeline(
    ConfigurableModule[T_ExplanationPipelineConfig],
    Generic[T_ExplanationPipelineConfig, T_TensorDataModel],
):
    __abstract__ = True
    __config__: type[T_ExplanationPipelineConfig]

    def __init__(
        self,
        config: T_ExplanationPipelineConfig,
        labels: DatasetLabels,
        persist_to_disk: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self._labels = labels
        self._persist_to_disk = persist_to_disk
        self._cache_dir = cache_dir
        self._build()
        if self._persist_to_disk and not self._cache_dir:
            raise ValueError("cache_dir must be specified if persist_to_disk is True.")

    @property
    def ops(self) -> Any:
        return ModelPipelineOps(self._model_pipeline)

    @property
    def cacher(self) -> ExplanationStateCacher | None:
        return self._cacher if self._persist_to_disk else None

    def summarize(self):
        logger.info("XAI Model Pipeline Summary:")
        logger.info(self._model_pipeline.ops.summarize())
        logger.info("Explainer Summary:")
        logger.info("Explainer: %s", self._explainer)
        logger.info("Feature Segmentor Config: %s", self.config.feature_segmentor)
        logger.info("Baseline Generator Config: %s", self.config.baseline_generator)

    def _dump_config(self, config_dir: Path) -> dict:
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "config.yaml", "w") as f:
            f.write(self._config.to_yaml())
            return self._config.model_dump()

    def _build_model_pipeline(self):
        self._model_pipeline = self.config.model_pipeline.build(labels=self._labels)

    def _build_explainer(self):
        # build model with wrapped forward
        self._model_signature = inspect.signature(self._model_pipeline._model.forward)
        self._wrapped_model = self._wrap_model_forward(self._model_pipeline._model)

        # build explainer
        self._explainer = self.config.explainer.build(
            model=self._wrapped_model,
            multi_target=False,
            internal_batch_size=self.config.internal_batch_size,
            grad_batch_size=self.config.grad_batch_size,
        )

        # get possible explainer args
        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        self._explainer_args = inspect.signature(
            self._explainer.explain
        ).parameters.keys()

    def _build_feature_segmentor(self):
        self._feature_segmentor = self.config.feature_segmentor.build()

    def _build_baseline_generator(self):
        self._baseline_generator = self.config.baseline_generator.build(
            model=self._model_pipeline._model
        )
        self._metric_baseline_generator = self.config.metric_baseline_generator.build(
            model=self._model_pipeline._model
        )

    def _build_cacher(self):
        self._explainer_dir = None
        if self._persist_to_disk:
            assert self._cache_dir is not None, (
                "cache_dir must be specified if persist_to_disk is True."
            )
            self._explainer_dir = (
                Path(self._cache_dir) / self._config.explainer.type.split("/")[-1]
            )
            self._dump_config(config_dir=self._explainer_dir)
            self._cacher = ExplanationStateCacher(
                cache_dir=self._explainer_dir, config=self.config
            )

            logger.info("Explanation caching enabled.")
            logger.info(f"Storing outputs to file = {self._cacher.file_path}")

    def _build(self):
        # build model pipeline
        self._build_model_pipeline()

        # build explainer
        self._build_explainer()

        # build feature segmentor
        self._build_feature_segmentor()

        # build baselines generator
        self._build_baseline_generator()

        # build cacher
        self._build_cacher()

    def _wrap_model_forward(self, model: torch.nn.Module) -> torch.nn.Module:
        class WrappedModel(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self._model = model

            def forward(self, *args: torch.Tensor) -> torch.Tensor:
                # we need to wrap the model like  this since in captum all args are passed as
                # *inputs + *additional_forward_args
                # this means we always need to make sure the input sequence is preserved
                from torch.nn.functional import softmax

                model_outputs = self._model(*args)
                if isinstance(model_outputs, dict):
                    logits = model_outputs["logits"]
                elif hasattr(model_outputs, "logits"):
                    logits = model_outputs.logits
                else:
                    logits = model_outputs
                return softmax(logits, dim=-1)

        return WrappedModel(model)

    @abstractmethod
    def _target(
        self, batch: T_TensorDataModel, model_outputs: Any
    ) -> BatchExplanationTarget | list[BatchExplanationTarget]:
        """Prepare the explanation target based on the strategy."""
        pass

    @abstractmethod
    def _explained_inputs(
        self, batch: T_TensorDataModel, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Prepare the input features for the explainer."""
        pass

    def _additional_forward_kwargs(
        self, batch: T_TensorDataModel
    ) -> dict[str, Any] | None:
        """Prepare any additional forward arguments for the explainer."""
        return None

    def _baselines(
        self, explained_inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Generate baselines for the explainer."""
        logger.debug(
            "Generating baselines using baseline generator with config: %s",
            self.config.baseline_generator,
        )
        baselines = self._baseline_generator(explained_inputs, **kwargs)
        log_tensor_info(baselines, name="baselines")
        return baselines

    def _metric_baselines(
        self, explained_inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Generate baselines for the explainer."""
        logger.debug(
            "Generating baselines using baseline generator with config: %s",
            self.config.metric_baseline_generator,
        )
        baselines = self._metric_baseline_generator(explained_inputs, **kwargs)
        log_tensor_info(baselines, name="metric_baselines")
        return baselines

    def _feature_mask(
        self, explained_inputs: dict[str, torch.Tensor], **kwargs
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor] | None]:
        """Generate feature mask using the feature segmentor."""
        logger.debug(
            "Generating feature mask using feature segmentor with config: %s",
            self.config.feature_segmentor,
        )
        feature_masks = self._feature_segmentor(explained_inputs, **kwargs)
        log_tensor_info(feature_masks, name="feature_masks")
        return feature_masks, None

    def _sliding_window_shapes_and_strides(
        self, input_feature_keys: tuple[str, ...]
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
            sliding_window_shapes_map[key] = self.config.sliding_window_shapes_map[key]
            strides[key] = self.config.strides_map[key]

        return sliding_window_shapes_map, strides

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

    def prepare_explanation_inputs(
        self, batch: T_TensorDataModel
    ) -> tuple[Any, BatchExplanationInputs]:
        """Prepare the inputs for the explainer step."""
        with torch.no_grad():
            # prepare explained inputs
            inputs = self._explained_inputs(batch)

            # prepare additional forward args
            additional_forward_kwargs = (
                self._additional_forward_kwargs(batch) or OrderedDict()
            )

            # prepare baselines
            baselines = self._baselines(explained_inputs=inputs)

            # prepare baselines for metrics if needed
            metric_baselines = None
            if self.config.explainability_metrics is not None:
                metric_baselines = self._metric_baselines(inputs)

            # prepare feature mask
            feature_mask, frozen_features = self._feature_mask(explained_inputs=inputs)

            # prepare sliding window shapes map and strides map for occlusion explainer
            sliding_window_shapes, strides = self._sliding_window_shapes_and_strides(
                input_feature_keys=tuple(inputs.keys())
            )

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

    def explainer_forward(
        self, explanation_inputs: BatchExplanationInputs
    ) -> tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]]:
        from torchxai.data_types import ExplanationTarget

        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        kwargs = {}
        for arg in self._explainer_args:
            kwargs[arg] = getattr(explanation_inputs, arg)

        def _map_target(
            target: BatchExplanationTarget | list[BatchExplanationTarget] | None,
        ) -> ExplanationTarget | list[ExplanationTarget]:
            if target is None:
                return ExplanationTarget.from_raw_input(None)
            if isinstance(target, BatchExplanationTarget):
                return ExplanationTarget.from_raw_input(target.value)
            elif isinstance(target, list):
                return [ExplanationTarget.from_raw_input(t.value) for t in target]
            else:
                raise ValueError(
                    "Target must be of type BatchExplanationTarget, list of BatchExplanationTarget, or None."
                )

        # map targets
        kwargs["target"] = _map_target(kwargs.pop("target", None))

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

        target = kwargs.pop("target", None)
        if self.config.iterative_computation and isinstance(target, list):
            logger.info(
                "Running explainer forward with iterative computation for multi-target explanations."
            )
            # disable multi-target for iterative computation
            self._explainer.multi_target = False

            batched = True
            if batched:
                # if the input is batched we need to repeat the inputs for each target and compute explanations in a single forward pass
                # Assumption: original batch size is always 1
                per_target_explanations = []
                internal_batch_size = self._explainer._internal_batch_size or len(
                    target
                )

                logger.info(
                    "internal_batch_size for iterative computation: %d",
                    internal_batch_size,
                )
                logger.info("Total number of targets: %d", len(target))

                # Calculate how many targets we can process at once
                # Since original batch size is 1, we can process internal_batch_size targets simultaneously
                num_targets_per_batch = internal_batch_size

                for batch_start in tqdm(
                    range(0, len(target), num_targets_per_batch),
                    desc="Computing explanations per target batch",
                ):
                    batch_end = min(batch_start + num_targets_per_batch, len(target))
                    target_batch = target[batch_start:batch_end]
                    target_batch = SingleTargetPerSample(
                        indices=[t.value[0] for t in target_batch]
                    )
                    num_targets_in_batch = len(target_batch.value)

                    # Repeat inputs for each target in the batch
                    batched_kwargs = {}
                    for key, value in kwargs.items():
                        if key == "inputs" and isinstance(value, tuple):
                            # Repeat each input tensor for each target
                            batched_kwargs[key] = tuple(
                                inp.repeat_interleave(num_targets_in_batch, dim=0)
                                for inp in value
                            )
                        elif key == "additional_forward_args" and isinstance(
                            value, tuple
                        ):
                            # Repeat additional forward args
                            batched_kwargs[key] = tuple(
                                arg.repeat_interleave(num_targets_in_batch, dim=0)
                                if isinstance(arg, torch.Tensor)
                                else arg
                                for arg in value
                            )
                        elif (
                            key in ["baselines", "feature_mask"]
                            and value is not None
                            and isinstance(value, tuple)
                        ):
                            # Repeat baselines and feature masks
                            batched_kwargs[key] = tuple(
                                item.repeat_interleave(num_targets_in_batch, dim=0)
                                for item in value
                            )
                        else:
                            # Keep other args as is
                            batched_kwargs[key] = value

                    # Compute explanations for the batch
                    curr_explanations = self._explainer.explain(
                        **batched_kwargs, target=target_batch
                    )
                    assert isinstance(curr_explanations, tuple), (
                        "Explainer returned invalid type during iterative computation. "
                        "Expected tuple."
                    )

                    # The results are organized as: [s0_t0, s0_t1, ..., s0_tT, s1_t0, s1_t1, ..., s1_tT, ...]
                    # We need to reorganize them per target: each target gets [s0_ti, s1_ti, ...]
                    for target_idx in range(num_targets_in_batch):
                        # Extract explanations for this target across all samples
                        # Every num_targets_in_batch-th element, starting from target_idx
                        target_explanation = tuple(
                            exp[target_idx::num_targets_in_batch].detach().cpu()
                            for exp in curr_explanations
                        )
                        per_target_explanations.append(target_explanation)
            else:
                per_target_explanations = []
                for t in tqdm(target, desc="Computing explanations per target"):
                    curr_explanations = self._explainer.explain(**kwargs, target=t)
                    assert isinstance(curr_explanations, tuple), (
                        "Explainer returned invalid type during iterative computation. "
                        "Expected tuple."
                    )
                    per_target_explanations.append(curr_explanations)

            # re-enable multi-target
            self._explainer.multi_target = True
            return per_target_explanations
        else:
            logger.info(
                "Running explainer forward with multi_target=%s",
                self._explainer.multi_target,
            )
            # we need to map the atria_insights target to torchxai target
            if isinstance(target, list):
                self._explainer.multi_target = True
            explanations = self._explainer.explain(**kwargs, target=target)

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

    def _validate_and_load_from_disk(
        self, explanation_inputs: BatchExplanationInputs, model_outputs: torch.Tensor
    ) -> ExplanationStepOutput:
        # load full batch from cache
        explanation_state = []
        for sample_id in explanation_inputs.sample_id:
            cached_state = self._cacher.load_sample(sample_id)
            explanation_state.append(cached_state)

        explanation_state = BatchExplanationState.fromlist(explanation_state)

        assert explanation_state.sample_id == explanation_inputs.sample_id, (
            "Sample IDs do not match between loaded explanation states and explanation inputs."
        )
        assert explanation_state.target == explanation_inputs.target, (
            "Targets do not match between loaded explanation states and explanation inputs."
            " Found "
            f"{explanation_state.target} =/= {explanation_inputs.target}"
        )
        assert explanation_state.feature_keys == explanation_inputs.feature_keys, (
            "Feature keys do not match between loaded explanation states and explanation inputs."
            f" Found {explanation_state.feature_keys} =/= {explanation_inputs.feature_keys}"
        )
        # assert (
        #     explanation_state.sliding_window_shapes
        #     == explanation_inputs.sliding_window_shapes
        # ), (
        #     "Sliding window shapes do not match between loaded explanation states and explanation inputs."
        # )
        # assert explanation_state.strides == explanation_inputs.strides, (
        #     "Strides do not match between loaded explanation states and explanation inputs."
        # )
        # if (
        #     explanation_state.feature_mask is not None
        #     and explanation_inputs.feature_mask is not None
        # ):
        #     fm1 = (fm.detach().cpu() for fm in explanation_state.feature_mask)
        #     fm2 = (fm.detach().cpu() for fm in explanation_inputs.feature_mask)
        #     assert all(torch.equal(a, b) for a, b in zip(fm1, fm2, strict=True)), (
        #         "Feature masks do not match between loaded explanation states and explanation inputs."
        #         f" Found {fm1} =/= {fm2}"
        #     )
        # if (
        #     explanation_state.frozen_features is not None
        #     and explanation_inputs.frozen_features is not None
        # ):
        #     f1 = [x.detach().cpu() for x in explanation_state.frozen_features]
        #     f2 = [x.detach().cpu() for x in explanation_inputs.frozen_features]
        #     assert all(torch.equal(a, b) for a, b in zip(f1, f2, strict=True)), (
        #         "Frozen features do not match between loaded explanation states and explanation inputs."
        #         f" Found {f1} =/= {f2}"
        #     )
        # assert (
        #     torch.mean(
        #         torch.abs(
        #             explanation_state.model_outputs.detach().cpu()
        #             - model_outputs.detach().cpu()
        #         )
        #     ).item()
        #     < 1e-3
        # ), (
        #     "Model outputs do not match between loaded explanation states and current model outputs."
        #     f"Found {model_outputs.detach().cpu()} =/= {explanation_state.model_outputs.detach().cpu()}"
        # )

        logger.info(
            "Loaded cached explanations for full batch of size %d.", len(model_outputs)
        )
        return ExplanationStepOutput(
            explanation_inputs=explanation_inputs,
            explanation_state=explanation_state.to_device(
                explanation_inputs.inputs[0].device
            ),
        )

    def explanation_step(self, batch: T_TensorDataModel) -> ExplanationStepOutput:
        # prepare explanation inputs
        model_outputs, explanation_inputs = self.prepare_explanation_inputs(batch=batch)

        if self._persist_to_disk:
            # check if full batch is already done
            is_batch_done = True
            for sample_id in explanation_inputs.sample_id:
                if not self._cacher.sample_exists(sample_id):
                    is_batch_done = False
                    break

            if is_batch_done:
                # load full batch from cache
                logger.debug(
                    f"Found cached explanations for full batch of size {len(batch)}. Loading from disk."
                )
                try:
                    return self._validate_and_load_from_disk(
                        explanation_inputs=explanation_inputs,
                        model_outputs=model_outputs,
                    )
                except Exception as e:
                    if self.config.throw_on_load_mismatch:
                        raise e
                    logger.warning(
                        f"Failed to validate loaded explanations due to error: {e}. Recomputing explanations."
                    )
                    logger.exception(e)

        # Track compute metrics if enabled
        compute_metrics = None
        if self.config.profile_time:
            device = (
                explanation_inputs.inputs[0].device
                if explanation_inputs.inputs
                else "cpu"
            )

            if torch.cuda.is_available() and "cuda" in str(device):
                # Use CUDA events for GPU timing
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()  # wait for previous work
                starter.record()

                # Run explanation
                explanations = self.explainer_forward(
                    explanation_inputs=explanation_inputs
                )

                ender.record()
                torch.cuda.synchronize()  # wait for events to finish

                elapsed_ms = starter.elapsed_time(ender)
            else:
                # Use time.perf_counter for CPU timing
                import time

                start_time = time.perf_counter()

                # Run explanation
                explanations = self.explainer_forward(
                    explanation_inputs=explanation_inputs
                )

                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000  # Convert to milliseconds

            # Create compute metrics
            compute_metrics = ComputeMetrics(
                elapsed_time_ms=elapsed_ms, device=str(device)
            )

            logger.info(
                f"Explanation computation took {elapsed_ms:.3f} ms on device {device}"
            )
        else:
            # Run explanation without timing
            explanations = self.explainer_forward(explanation_inputs=explanation_inputs)

        assert explanation_inputs.feature_keys is not None, "feature_keys must be set."

        # prepare explanation states
        explanation_state = BatchExplanationState(
            sample_id=explanation_inputs.sample_id,
            target=explanation_inputs.target,
            attention_token_target=explanation_inputs.attention_token_target,
            feature_keys=explanation_inputs.feature_keys,
            frozen_features=explanation_inputs.frozen_features,
            sliding_window_shapes=explanation_inputs.sliding_window_shapes,
            strides=explanation_inputs.strides,
            feature_mask=explanation_inputs.feature_mask,
            model_outputs=model_outputs,
            explanations=MultiTargetBatchExplanation(
                value=[BatchExplanation(value=exp) for exp in explanations]
            )
            if isinstance(explanations, list)
            else BatchExplanation(value=explanations),
            compute_metrics=compute_metrics,
        )

        # save to disk
        if self._persist_to_disk:
            for sample_explanation_state in explanation_state.tolist():
                self._cacher.save_sample(sample_explanation_state)

        return ExplanationStepOutput(
            explanation_inputs=explanation_inputs, explanation_state=explanation_state
        )

    def build_metrics(self, device: torch.device | str = "cpu") -> dict[str, Metric]:
        if self.config.explainability_metrics is None:
            return {}

        # build explainer
        x_metrics = {}
        for key, value in self.config.explainability_metrics.items():
            logger.info(
                "Building explainability metric '%s' with config: %s", key, value
            )
            x_metrics[key] = value.build(
                model=self._wrapped_model,
                explainer=self._explainer,
                device=device,
                persist_to_disk=self._persist_to_disk,
                cache_dir=self._explainer_dir,
                metric_name=key,
            )
        return x_metrics
