from typing import Any, ClassVar, Literal

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
    max_features_processed_per_batch: int | None = 10
    total_feature_bins: int = 100
    n_random_perms: int = 10
    seed: int | None = None
    show_progress: bool = True
    return_intermediate_results: bool = True
    __hash_exclude__: ClassVar[set[str]] = {
        "enabled",
        "max_features_processed_per_batch",
        "show_progress",
        "return_intermediate_results",
    }

    @property
    def name(self):
        seed_part = f".seed_{self.seed}" if self.seed is not None else ""
        return (
            f"aopc"
            f".tfb_{self.total_feature_bins}"
            f".nrp_{self.n_random_perms}"
            f"{seed_part}"
        )


class AOPC(ExplainabilityMetric[AOPCConfig]):
    __config__ = AOPCConfig

    def compute(self) -> dict:
        if not self._results:
            return {}

        import numpy as np

        def _resample(arr, n=101):
            arr = np.asarray(arr, dtype=float)
            m = len(arr)
            if m < 2:
                return np.full(n, arr[0])
            return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, m), arr)

        def gather_curves(key, reduce_perms=False):
            curves = []
            for batch in self._results:
                if key not in batch:
                    continue
                v = batch[key].cpu().float().numpy()
                if reduce_perms:
                    # [B, n_perms, n_bins] or [B, n_targets, n_perms, n_bins]
                    v = v.mean(axis=-2)
                # reshape to [N, n_bins], flattening batch and any target dims
                v = v.reshape(-1, v.shape[-1])
                for sample_curve in v:
                    if not np.isnan(sample_curve).any():
                        curves.append(_resample(sample_curve))
            return np.stack(curves).mean(0) if curves else np.full(101, float("nan"))

        desc_mean = gather_curves("desc")
        asc_mean = gather_curves("asc")
        rand_mean = gather_curves("rand", reduce_perms=True)

        exec_times = torch.cat(
            [batch["sample_exec_time"].flatten().float() for batch in self._results]
        )
        return {
            "desc": desc_mean.tolist(),
            "asc": asc_mean.tolist(),
            "rand": rand_mean.tolist(),
            "abpc": float((desc_mean - asc_mean)[-1]),
            "desc_minus_rand": float((desc_mean - rand_mean)[-1]),
            "asc_minus_rand": float((asc_mean - rand_mean)[-1]),
            "exec_time": torch.nanmean(exec_times).item(),
        }

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
            baselines=explanation_inputs.metric_baselines,
            feature_mask=explanation_inputs.metric_feature_mask,
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            max_features_processed_per_batch=self.config.max_features_processed_per_batch,  # type: ignore
            total_feature_bins=self.config.total_feature_bins,
            n_random_perms=self.config.n_random_perms,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
            multi_target=explanation_inputs.is_multi_target,
            return_intermediate_results=True,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/faithfulness_correlation")
class FaithfulnessCorrelationConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/faithfulness_correlation"] = (  # type: ignore
        "faithfulness/faithfulness_correlation"
    )
    perturb_func: str = "fixed"
    n_perturb_samples: int = 10
    max_examples_per_batch: int | None = 10
    percent_features_perturbed: float = 0.1
    show_progress: bool = True
    return_intermediate_results: bool = True
    __hash_exclude__: ClassVar[set[str]] = {
        "enabled",
        "max_examples_per_batch",
        "show_progress",
        "return_intermediate_results",
    }

    @property
    def name(self):
        return (
            f"faithfulness_correlation"
            f".pf_{self.perturb_func}"
            f".nps_{self.n_perturb_samples}"
            f".pfp_{self.percent_features_perturbed}"
        )


class FaithfulnessCorrelation(ExplainabilityMetric[FaithfulnessCorrelationConfig]):
    __config__ = FaithfulnessCorrelationConfig
    _score_keys: ClassVar[list[str]] = ["faithfulness_corr_score"]

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
            baselines=explanation_inputs.metric_baselines,
            feature_mask=explanation_inputs.metric_feature_mask,
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            perturb_func=perturb_func,
            n_perturb_samples=self.config.n_perturb_samples,
            max_examples_per_batch=self.config.max_examples_per_batch,
            percent_features_perturbed=self.config.percent_features_perturbed,
            show_progress=self.config.show_progress,
            multi_target=explanation_inputs.is_multi_target,
            return_intermediate_results=True,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/faithfulness_estimate")
class FaithfulnessEstimateConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/faithfulness_estimate"] = (  # type: ignore
        "faithfulness/faithfulness_estimate"
    )
    max_features_processed_per_batch: int | None = 10
    percentage_feature_removal_per_step: float = 0.0
    show_progress: bool = True
    return_intermediate_results: bool = True
    __hash_exclude__: ClassVar[set[str]] = {
        "enabled",
        "max_features_processed_per_batch",
        "show_progress",
        "return_intermediate_results",
    }

    @property
    def name(self):
        return f"faithfulness_estimate.pfrs_{self.percentage_feature_removal_per_step}"


class FaithfulnessEstimate(ExplainabilityMetric[FaithfulnessEstimateConfig]):
    __config__ = FaithfulnessEstimateConfig
    _score_keys: ClassVar[list[str]] = ["faithfulness_estimate_score"]

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
            baselines=explanation_inputs.metric_baselines,
            feature_mask=explanation_inputs.metric_feature_mask,
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            max_features_processed_per_batch=self.config.max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self.config.percentage_feature_removal_per_step,
            multi_target=explanation_inputs.is_multi_target,
            show_progress=self.config.show_progress,
            return_intermediate_results=True,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/infidelity")
class InfidelityConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/infidelity"] = (  # type: ignore
        "faithfulness/infidelity"
    )
    perturb_func: str = "default_infidelity_perturb_func"
    perturbation_noise_scale: float = 0.003
    n_perturb_samples: int = 10
    max_examples_per_batch: int | None = None
    normalize: bool = True
    __hash_exclude__: ClassVar[set[str]] = {"enabled", "max_examples_per_batch"}

    @property
    def name(self):
        return (
            f"infidelity"
            f".pf_{self.perturb_func}"
            f".pns_{self.perturbation_noise_scale}"
            f".nps_{self.n_perturb_samples}"
            f".norm_{int(self.normalize)}"
        )


class Infidelity(ExplainabilityMetric[InfidelityConfig]):
    __config__ = InfidelityConfig
    _score_keys: ClassVar[list[str]] = ["infidelity_score"]

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
            baselines=explanation_inputs.metric_baselines,
            feature_mask=explanation_inputs.metric_feature_mask,
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
    max_features_processed_per_batch: int | None = None
    percentage_feature_removal_per_step: float = 0.01
    show_progress: bool = True
    return_intermediate_results: bool = True
    __hash_exclude__: ClassVar[set[str]] = {
        "enabled",
        "max_features_processed_per_batch",
        "show_progress",
        "return_intermediate_results",
    }

    @property
    def name(self):
        return f"monotonicity.pfrs_{self.percentage_feature_removal_per_step}"


class Monotonicity(ExplainabilityMetric[MonotonicityConfig]):
    __config__ = MonotonicityConfig
    _score_keys: ClassVar[list[str]] = ["monotonicity_score"]

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
            baselines=explanation_inputs.metric_baselines,
            feature_mask=explanation_inputs.metric_feature_mask,
            target=self._map_target(explanation_inputs.target),
            frozen_features=explanation_inputs.frozen_features,
            max_features_processed_per_batch=self.config.max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self.config.percentage_feature_removal_per_step,
            multi_target=explanation_inputs.is_multi_target,
            show_progress=self.config.show_progress,
            return_intermediate_results=True,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("faithfulness/sensitivity_n")
class SensitivityNConfig(ExplainabilityMetricConfig):
    type: Literal["faithfulness/sensitivity_n"] = (  # type: ignore
        "faithfulness/sensitivity_n"
    )
    n_features_perturbed: int | float = 10
    n_perturb_samples: int = 10
    max_examples_per_batch: int | None = None
    normalize: bool = False
    __hash_exclude__: ClassVar[set[str]] = {"enabled", "max_examples_per_batch"}

    @property
    def name(self):
        return (
            f"sensitivity_n"
            f".nfp_{self.n_features_perturbed}"
            f".nps_{self.n_perturb_samples}"
            f".norm_{int(self.normalize)}"
        )


class SensitivityN(ExplainabilityMetric[SensitivityNConfig]):
    __config__ = SensitivityNConfig
    _score_keys: ClassVar[list[str]] = ["sensitivity_n_score"]

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
            baselines=explanation_inputs.metric_baselines,
            feature_mask=explanation_inputs.metric_feature_mask,
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
