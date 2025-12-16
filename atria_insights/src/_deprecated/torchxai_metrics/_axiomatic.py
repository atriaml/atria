from typing import Any

import torch
from torch.nn.modules import Module
from torchxai.explainers import Explainer
from torchxai.metrics.axiomatic.completeness import completeness
from torchxai.metrics.axiomatic.input_invariance import input_invariance
from torchxai.metrics.axiomatic.monotonicity_corr_and_non_sens import (
    monotonicity_corr_and_non_sens,
)

from atria_insights.core.data_types import ModelExplainerOutput
from atria_insights.core.explainability_metrics.torchxai_metrics._base import (
    TorchXAIMetricBase,
)
from atria_insights.core.explainability_metrics.torchxai_metrics._utilities import (
    ModelExplainerOutputTransform,
)


class CompletenessMetric(TorchXAIMetricBase):
    def _update(
        self, model_output: ModelExplainerOutput, is_multi_target: bool = False
    ) -> Any:
        transform = ModelExplainerOutputTransform(model_output)
        score = completeness(
            forward_func=self._model,
            inputs=transform.inputs,
            attributions=transform.attributions,
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=transform.metric_baselines,
            additional_forward_args=transform.additional_forward_args,
            target=transform.target,
            is_multi_target=is_multi_target,
            return_dict=True,
        )
        self._results.append(score)
        return score


class InputInvarianceMetric(TorchXAIMetricBase):
    def __init__(
        self, model: Module, explainer: Explainer, with_amp: bool = False, device="cpu"
    ):
        self._explainer = explainer
        assert self._explainer.model is model, (
            "Explainer model and metric model must be the same"
        )
        super().__init__(model, with_amp, device)

    def _update(
        self, model_output: ModelExplainerOutput, is_multi_target: bool = False
    ) -> Any:
        transform = ModelExplainerOutputTransform(model_output)
        assert transform.constant_shifts is not None, (
            "Constant shifts must be provided for input invariance metric"
        )
        assert transform.input_layer_names is not None, (
            "Input layer names must be provided for input invariance metric"
        )
        score = input_invariance(
            explainer=self._explainer,
            inputs=transform.inputs,
            constant_shifts=transform.constant_shifts,
            input_layer_names=transform.input_layer_names,  # type: ignore
            return_intermediate_results=False,
            return_dict=True,
            # these are additionall explainer forward call args
            # NOTE:
            # notice explainer baselines here
            # this is used to compute attributions on the go during metric computation
            # this metric does not use metric baselines
            targets=transform.target,
            baselines=transform.explainer_baselines,  # notice explainer baselines, this is different from metric baselines
            feature_mask=transform.feature_mask,
            additional_forward_args=transform.additional_forward_args,
        )
        self._results.append(score)
        return score


class MonotonicityCorrAndNonSensMetric(TorchXAIMetricBase):
    def __init__(
        self,
        model: torch.nn.Module,
        with_amp: bool = False,
        device="cpu",
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_batch: int | None = None,
        percentage_feature_removal_per_step: float = 0,
        zero_attribution_threshold: float = 0.00001,
        zero_variance_threshold: float = 0.00001,
        use_percentage_attribution_threshold: bool = False,
    ):
        self._n_perturbations_per_feature = n_perturbations_per_feature
        self._max_features_processed_per_batch = max_features_processed_per_batch
        self._percentage_feature_removal_per_step = percentage_feature_removal_per_step
        self._zero_attribution_threshold = zero_attribution_threshold
        self._zero_variance_threshold = zero_variance_threshold
        self._use_percentage_attribution_threshold = (
            use_percentage_attribution_threshold
        )
        self._num_examples = 0

        super().__init__(model=model, with_amp=with_amp, device=device)

    def _update(
        self, model_output: ModelExplainerOutput, is_multi_target: bool = False
    ) -> Any:
        transform = ModelExplainerOutputTransform(model_output)
        return monotonicity_corr_and_non_sens(
            forward_func=self._model,
            inputs=transform.inputs,
            attributions=transform.attributions,
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=transform.metric_baselines,
            feature_mask=transform.feature_mask,
            additional_forward_args=transform.additional_forward_args,
            target=transform.target,
            frozen_features=transform.frozen_features,
            # perturb_func: (...) -> Unknown = default_fixed_baseline_perturb_func(),
            n_perturbations_per_feature=self._n_perturbations_per_feature,
            max_features_processed_per_batch=self._max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self._percentage_feature_removal_per_step,
            zero_attribution_threshold=self._zero_attribution_threshold,
            zero_variance_threshold=self._zero_variance_threshold,
            use_percentage_attribution_threshold=self._use_percentage_attribution_threshold,
            return_ratio=True,
            show_progress=False,
            return_intermediate_results=False,
            is_multi_target=is_multi_target,
        )
