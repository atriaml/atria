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
from tqdm import tqdm

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.data_types._explanation_state import (
    BatchExplanation,
    BatchExplanationState,
)
from atria_insights.data_types._targets import BatchExplanationTarget
from atria_insights.engines._explanation_step import ExplanationStepOutput
from atria_insights.model_pipelines._common import (
    ExplanationTargetStrategy,
    T_ExplainableModelPipelineConfig,
)
from atria_insights.storage.sample_cache_managers._explanation_state import (
    ExplanationStateCacher,
)

logger = get_logger(__name__)

_DEFAULT_FEATURE_INPUT_KEY = "input_feature"


class ExplainableModelPipeline(
    ConfigurableModule[T_ExplainableModelPipelineConfig],
    Generic[T_ExplainableModelPipelineConfig, T_TensorDataModel],
):
    __abstract__ = True
    __config__: type[T_ExplainableModelPipelineConfig]

    def __init__(
        self,
        config: T_ExplainableModelPipelineConfig,
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
        # build explainer
        multi_target = (
            self.config.explanation_target_strategy == ExplanationTargetStrategy.all
        )

        # build model with wrapped forward
        self._model_signature = inspect.signature(self._model_pipeline._model.forward)
        self._wrapped_model = self._wrap_model_forward(self._model_pipeline._model)

        # build explainer
        self._explainer = self.config.explainer.build(
            model=self._wrapped_model,
            multi_target=multi_target,
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
        self, batch: T_TensorDataModel
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        """Prepare the input features for the explainer."""
        pass

    def _additional_forward_kwargs(
        self, batch: T_TensorDataModel
    ) -> OrderedDict[str, Any] | None:
        """Prepare any additional forward arguments for the explainer."""
        return None

    def _baselines(
        self, explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        """Generate baselines for the explainer."""
        logger.debug(
            "Generating baselines using baseline generator with config: %s",
            self.config.baseline_generator,
        )
        baselines = self._baseline_generator(explained_inputs, **kwargs)
        log_tensor_info(baselines, name="baselines")
        return baselines

    def _feature_mask(
        self, explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor], **kwargs
    ) -> Any:
        """Generate feature mask using the feature segmentor."""
        logger.debug(
            "Generating feature mask using feature segmentor with config: %s",
            self.config.feature_segmentor,
        )
        feature_masks = self._feature_segmentor(explained_inputs, **kwargs)
        log_tensor_info(feature_masks, name="feature_masks")
        return feature_masks

    def _validated_inputs(
        self,
        inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
        additional_forward_kwargs: OrderedDict[str, Any] | None = None,
        baselines: torch.Tensor | OrderedDict[str, torch.Tensor] | None = None,
        feature_mask: torch.Tensor | OrderedDict[str, torch.Tensor] | None = None,
    ) -> tuple:
        """
        Validate and map inputs to the model forward signature.

        Returns:
            model_inputs: tuple of positional arguments for model forward
            expected_params: list of expected parameter names (excluding self)
        """
        additional_forward_kwargs = additional_forward_kwargs or OrderedDict()

        # ---- model signature ----
        expected_params = list(self._model_signature.parameters.keys())

        # ---- inputs ----
        baselines_tuple = None
        feature_mask_tuple = None
        if isinstance(inputs, OrderedDict):
            input_names = list(inputs.keys())
            input_values = tuple(inputs.values())
            if baselines is not None:
                assert isinstance(baselines, OrderedDict), (
                    "If inputs is an OrderedDict, baselines must also be an OrderedDict."
                )
                baselines_tuple = tuple(baselines[key] for key in input_names)
            if feature_mask is not None:
                assert isinstance(feature_mask, OrderedDict), (
                    "If inputs is an OrderedDict, feature_mask must also be an OrderedDict."
                )
                feature_mask_tuple = tuple(feature_mask[key] for key in input_names)
        else:
            input_names = [expected_params[0]]
            input_values = (inputs,)
            if baselines is not None:
                assert isinstance(baselines, torch.Tensor), (
                    "If inputs is a Tensor, baselines must also be a Tensor."
                )
                baselines_tuple = (baselines,)
            if feature_mask is not None:
                assert isinstance(feature_mask, torch.Tensor), (
                    "If inputs is a Tensor, feature_mask must also be a Tensor."
                )
                feature_mask_tuple = (feature_mask,)

        # ---- additional kwargs ----
        additional_names = list(additional_forward_kwargs.keys())
        additional_values = tuple(additional_forward_kwargs.values())

        # ---- combined ----
        all_names = input_names + additional_names

        # ---- validation ----
        if len(all_names) != len(expected_params):
            raise ValueError(
                f"Model expects {len(expected_params)} inputs {expected_params}, "
                f"but got {len(all_names)} inputs {all_names}."
            )

        for given, expected in zip(all_names, expected_params, strict=True):
            if given != expected:
                raise ValueError(
                    f"Input '{given}' does not match model parameter '{expected}'.\n"
                    f"Given inputs: {all_names}\n"
                    f"Expected signature: {expected_params}"
                )

        return input_values, additional_values, baselines_tuple, feature_mask_tuple

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

            # prepare feature mask
            feature_mask = self._feature_mask(explained_inputs=inputs)

            # map the inputs and forwad args to model signautre
            input_feature_keys = (
                tuple(inputs.keys())
                if isinstance(inputs, OrderedDict)
                else (_DEFAULT_FEATURE_INPUT_KEY,)  # make a dummy input
            )
            inputs, additional_forward_args, baselines, feature_mask = (
                self._validated_inputs(
                    inputs=inputs,
                    additional_forward_kwargs=additional_forward_kwargs,
                    baselines=baselines,
                    feature_mask=feature_mask,
                )
            )
            assert len(inputs) == len(input_feature_keys), (
                "Input feature keys length does not match inputs length."
                f" {len(input_feature_keys)=}, {len(inputs)=}"
            )

            # forward pass
            model_outputs = self._wrapped_model(*(*inputs, *additional_forward_args))

            # prepare target
            target = self._target(batch=batch, model_outputs=model_outputs)

            # prepare explanation inputs
            return model_outputs, BatchExplanationInputs(
                sample_id=batch.metadata.sample_id,
                inputs=inputs,
                additional_forward_args=additional_forward_args,
                baselines=baselines if "baselines" in self._explainer_args else None,
                feature_mask=feature_mask
                if "feature_mask" in self._explainer_args
                else None,
                target=target,
                frozen_features=None,
                feature_keys=input_feature_keys,
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

        logger.debug("Explainer forward with args:")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                logger.debug(
                    f"  {k}: Tensor shape {v.shape}, dtype {v.dtype}, device {v.device}"
                )
            elif isinstance(v, tuple):
                for item in v:
                    logger.debug(
                        f"  {k}: Tensor shape {item.shape}, dtype {item.dtype}, device {item.device}"
                    )
            else:
                logger.debug(f"  {k}: {type(v)}")

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
        target = _map_target(kwargs.pop("target", None))

        if self.config.iterative_computation and isinstance(target, list):
            # disable multi-target for iterative computation
            self._explainer.multi_target = False

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
            # we need to map the atria_insights target to torchxai target
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
        )
        assert (
            explanation_state.frozen_features == explanation_inputs.frozen_features
        ), (
            "Frozen features do not match between loaded explanation states and explanation inputs."
        )
        assert (
            torch.mean(
                torch.abs(
                    explanation_state.model_outputs.detach().cpu()
                    - model_outputs.detach().cpu()
                )
            ).item()
            < 1e-3
        ), (
            "Model outputs do not match between loaded explanation states and current model outputs."
            f"Found {model_outputs.detach().cpu()} =/= {explanation_state.model_outputs.detach().cpu()}"
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
                    logger.warning(
                        f"Failed to validate loaded explanations due to error: {e}. Recomputing explanations."
                    )

        explanations = self.explainer_forward(explanation_inputs=explanation_inputs)
        assert explanation_inputs.feature_keys is not None, "feature_keys must be set."

        # prepare explanation states
        explanation_state = BatchExplanationState(
            sample_id=explanation_inputs.sample_id,
            target=explanation_inputs.target,
            feature_keys=explanation_inputs.feature_keys,
            frozen_features=explanation_inputs.frozen_features,
            model_outputs=model_outputs,
            explanations=BatchExplanation(value=explanations),
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
            logger.debug(
                "Building explainability metric '%s' with config: %s", key, value
            )
            x_metrics[key] = value.build(
                model=self._wrapped_model,
                explainer=self._explainer,
                device=device,
                persist_to_disk=self._persist_to_disk,
                cache_dir=self._explainer_dir,
            )
        return x_metrics
