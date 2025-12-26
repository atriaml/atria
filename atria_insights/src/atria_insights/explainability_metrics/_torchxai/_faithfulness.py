from typing import Any, Literal

import torch
from torchxai.metrics import (
    aopc,
    faithfulness_corr,
    faithfulness_estimate,
    infidelity,
    monotonicity,
    sensitivity_n,
)

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainability_metrics._torchxai._base import ExplainabilityMetric


@EXPLAINABILITY_METRICS.register("faithfulness/aopc")
class AOPCConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/aopc"] = "faithfulness/aopc"  # type: ignore
    module_path: str | None = "atria_insights.explainability_metrics.AOPC"
    max_features_processed_per_batch: int | None = 10
    total_feature_bins: int = 100
    n_random_perms: int = 10
    seed: int | None = None
    show_progress: bool = False
    return_intermediate_results: bool = False


class AOPC(ExplainabilityMetric[AOPCConfig]):
    __config__ = AOPCConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        outputs = aopc(
            forward_func=self._model,
            inputs=explanation_inputs.inputs,
            additional_forward_args=explanation_inputs.additional_forward_args,
            attributions=explanations,  # type: ignore
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=self._prepare_baselines(explanation_inputs),
            feature_mask=self._prepare_feature_mask(explanation_inputs),
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            max_features_processed_per_batch=self.config.max_features_processed_per_batch,  # type: ignore
            total_feature_bins=self.config.total_feature_bins,
            n_random_perms=self.config.n_random_perms,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
            return_intermediate_results=self.config.return_intermediate_results,
            multi_target=explanation_inputs.is_multi_target,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/faithfulness_correlation")
class FaithfulnessCorrelationConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/faithfulness_correlation"] = (  # type: ignore
        "faithfulness/faithfulness_correlation"
    )
    module_path: str | None = (
        "atria_insights.explainability_metrics.FaithfulnessCorrelation"
    )
    perturb_func: str = "fixed"
    n_perturb_samples: int = 10
    max_examples_per_batch: int | None = 10
    percent_features_perturbed: float = 0.1
    show_progress: bool = False
    return_intermediate_results: bool = False


class FaithfulnessCorrelation(ExplainabilityMetric[FaithfulnessCorrelationConfig]):
    __config__ = FaithfulnessCorrelationConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        from torchxai.metrics.complexity.effective_complexity import (
            default_fixed_baseline_perturb_func,
        )

        if self.config.perturb_func == "fixed":
            perturb_func = default_fixed_baseline_perturb_func()
        else:
            raise ValueError(
                f"Unsupported perturbation function: {self.config.perturb_func}"
            )
        outputs = faithfulness_corr(
            forward_func=self._model,
            inputs=explanation_inputs.inputs,
            additional_forward_args=explanation_inputs.additional_forward_args,
            attributions=explanations,  # type: ignore
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=self._prepare_baselines(explanation_inputs),
            feature_mask=self._prepare_feature_mask(explanation_inputs),
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            perturb_func=perturb_func,
            n_perturb_samples=self.config.n_perturb_samples,
            max_examples_per_batch=self.config.max_examples_per_batch,
            percent_features_perturbed=self.config.percent_features_perturbed,
            show_progress=self.config.show_progress,
            multi_target=explanation_inputs.is_multi_target,
            return_intermediate_results=self.config.return_intermediate_results,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/faithfulness_estimate")
class FaithfulnessEstimateConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/faithfulness_estimate"] = (  # type: ignore
        "faithfulness/faithfulness_estimate"
    )
    module_path: str | None = (
        "atria_insights.explainability_metrics.FaithfulnessEstimate"
    )
    max_features_processed_per_batch: int | None = 10
    percentage_feature_removal_per_step: float = 0.0
    show_progress: bool = False
    return_intermediate_results: bool = False


class FaithfulnessEstimate(ExplainabilityMetric[FaithfulnessEstimateConfig]):
    __config__ = FaithfulnessEstimateConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        outputs = faithfulness_estimate(
            forward_func=self._model,
            inputs=explanation_inputs.inputs,
            additional_forward_args=explanation_inputs.additional_forward_args,
            attributions=explanations,  # type: ignore
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=self._prepare_baselines(explanation_inputs),
            feature_mask=self._prepare_feature_mask(explanation_inputs),
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            max_features_processed_per_batch=self.config.max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self.config.percentage_feature_removal_per_step,
            multi_target=explanation_inputs.is_multi_target,
            show_progress=self.config.show_progress,
            return_intermediate_results=self.config.return_intermediate_results,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/infidelity")
class InfidelityConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/infidelity"] = (  # type: ignore
        "faithfulness/infidelity"
    )
    module_path: str | None = "atria_insights.explainability_metrics.Infidelity"
    perturb_func: str = "default_infidelity_perturb_func"
    perturbation_noise_scale: float = 0.003
    n_perturb_samples: int = 10
    max_examples_per_batch: int | None = None
    normalize: bool = True


class Infidelity(ExplainabilityMetric[InfidelityConfig]):
    __config__ = InfidelityConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        from torchxai.metrics.faithfulness.infidelity import (
            default_infidelity_perturb_fn,
        )

        outputs = infidelity(
            forward_func=self._model,
            inputs=explanation_inputs.inputs,
            additional_forward_args=explanation_inputs.additional_forward_args,
            attributions=explanations,  # type: ignore
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=self._prepare_baselines(explanation_inputs),
            feature_mask=self._prepare_feature_mask(explanation_inputs),
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            perturb_func=default_infidelity_perturb_fn(
                self.config.perturbation_noise_scale
            ),
            n_perturb_samples=self.config.n_perturb_samples,
            max_examples_per_batch=self.config.max_examples_per_batch,
            normalize=self.config.normalize,
            multi_target=explanation_inputs.is_multi_target,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/monotonicity")
class MonotonicityConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/monotonicity"] = (  # type: ignore
        "faithfulness/monotonicity"
    )
    module_path: str | None = "atria_insights.explainability_metrics.Monotonicity"
    max_features_processed_per_batch: int | None = None
    percentage_feature_removal_per_step: float = 0.01
    show_progress: bool = False
    return_intermediate_results: bool = False


class Monotonicity(ExplainabilityMetric[MonotonicityConfig]):
    __config__ = MonotonicityConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        outputs = monotonicity(
            forward_func=self._model,
            inputs=explanation_inputs.inputs,
            additional_forward_args=explanation_inputs.additional_forward_args,
            attributions=explanations,  # type: ignore
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=self._prepare_baselines(explanation_inputs),
            feature_mask=self._prepare_feature_mask(explanation_inputs),
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            max_features_processed_per_batch=self.config.max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self.config.percentage_feature_removal_per_step,
            multi_target=explanation_inputs.is_multi_target,
            show_progress=self.config.show_progress,
            return_intermediate_results=self.config.return_intermediate_results,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/sensitivity_n")
class SensitivityNConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/sensitivity_n"] = (  # type: ignore
        "faithfulness/sensitivity_n"
    )
    module_path: str | None = "atria_insights.explainability_metrics.SensitivityN"
    n_features_perturbed: int = 10
    n_perturb_samples: int = 10
    max_examples_per_batch: int | None = None
    normalize: bool = False


class SensitivityN(ExplainabilityMetric[SensitivityNConfig]):
    __config__ = SensitivityNConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        outputs = sensitivity_n(
            n_features_perturbed=self.config.n_features_perturbed,
            forward_func=self._model,
            inputs=explanation_inputs.inputs,
            additional_forward_args=explanation_inputs.additional_forward_args,
            attributions=explanations,  # type: ignore
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=self._prepare_baselines(explanation_inputs),
            feature_mask=self._prepare_feature_mask(explanation_inputs),
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            n_perturb_samples=self.config.n_perturb_samples,
            max_examples_per_batch=self.config.max_examples_per_batch,
            normalize=self.config.normalize,
            multi_target=explanation_inputs.is_multi_target,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs
