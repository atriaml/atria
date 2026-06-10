from collections import defaultdict
from typing import Any, ClassVar, Literal

import numpy as np
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
from atria_insights.explainability_metrics._torchxai._base import (
    BatchMetricOuptut,
    ExplainabilityMetric,
    MultiTargetBatchMetricOuptut,
)


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
            f"aopc.tfb_{self.total_feature_bins}.nrp_{self.n_random_perms}{seed_part}"
        )


class AOPC(ExplainabilityMetric[AOPCConfig]):
    __config__ = AOPCConfig
    _score_keys: ClassVar[list[str]] = [
        "desc_minus_rand",
        "abpc",
        "aopc_desc",
        "aopc_asc",
        "aopc_rand",
    ]

    def _resample_batch(
        self, v: torch.Tensor | list, reduce_perm: bool = False
    ) -> list[list[float]]:
        import numpy as np

        samples = v if isinstance(v, list) else [v[i] for i in range(v.shape[0])]
        result = []
        for s in samples:
            arr = (
                s.cpu().float().numpy()
                if isinstance(s, torch.Tensor)
                else np.asarray(s, dtype=float)
            )
            if reduce_perm and arr.ndim > 1:
                arr = arr.mean(axis=-2)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            m = len(arr)
            if m < 2:
                resampled: list[float] = [
                    float(arr[0]) if m > 0 else float("nan")
                ] * 101
            else:
                resampled = np.interp(
                    np.linspace(0, 1, 101), np.linspace(0, 1, m), arr.astype(float)
                ).tolist()
            result.append(resampled)
        return torch.tensor(result)

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

        with torch.no_grad():
            if explanation_inputs.is_multi_target:
                desc_per_target = outputs.pop("desc")
                asc_per_target = outputs.pop("asc")
                rand_per_target = outputs.pop("rand")
                print("desc_per_target,", desc_per_target)

                desc_r_per_target = torch.stack(
                    [self._resample_batch(desc) for desc in desc_per_target]
                )
                asc_r_per_target = torch.stack(
                    [self._resample_batch(asc) for asc in asc_per_target]
                )
                rand_r_per_target = torch.stack(
                    [
                        self._resample_batch(rand, reduce_perm=True)
                        for rand in rand_per_target
                    ]
                )

                assert len(desc_r_per_target) == len(asc_r_per_target), (
                    f"Lengths must match, found {len(desc_r_per_target)} =/= {len(asc_r_per_target)}"
                )
                assert len(rand_r_per_target) == len(asc_r_per_target), (
                    f"Lengths must match, found {len(rand_r_per_target)} =/= {len(asc_r_per_target)}"
                )

                aopc_per_target = [
                    [(desc_r[i][-1] - rand_r[i][-1]).item() for i in range(len(desc_r))]
                    for desc_r, rand_r in zip(
                        desc_r_per_target, rand_r_per_target, strict=True
                    )
                ]

                abpc_per_target = [
                    [(desc_r[i][-1] - asc_r[i][-1]).item() for i in range(len(desc_r))]
                    for desc_r, asc_r in zip(
                        desc_r_per_target, asc_r_per_target, strict=True
                    )
                ]

                return [
                    MultiTargetBatchMetricOuptut(
                        key="faithfulness/aopc_desc", value=desc_r_per_target.tolist()
                    ),
                    MultiTargetBatchMetricOuptut(
                        key="faithfulness/aopc_asc", value=asc_r_per_target.tolist()
                    ),
                    MultiTargetBatchMetricOuptut(
                        key="faithfulness/aopc_rand", value=rand_r_per_target.tolist()
                    ),
                    MultiTargetBatchMetricOuptut(
                        key="faithfulness/aopc", value=aopc_per_target
                    ),
                    MultiTargetBatchMetricOuptut(
                        key="faithfulness/abpc", value=abpc_per_target
                    ),
                ]

            else:
                desc = outputs.pop("desc")
                asc = outputs.pop("asc")
                rand = outputs.pop("rand")

                desc_r = self._resample_batch(desc)
                asc_r = self._resample_batch(asc)
                rand_r = self._resample_batch(rand, reduce_perm=True)

                batch_size = len(desc_r)
                return [
                    BatchMetricOuptut(
                        key="faithfulness/aopc_desc", value=desc_r.tolist()
                    ),
                    BatchMetricOuptut(
                        key="faithfulness/aopc_asc", value=asc_r.tolist()
                    ),
                    BatchMetricOuptut(
                        key="faithfulness/aopc_rand", value=rand_r.tolist()
                    ),
                    BatchMetricOuptut(
                        key="faithfulness/aopc",
                        value=[
                            desc_r[i][-1] - rand_r[i][-1] for i in range(batch_size)
                        ],
                    ),
                    BatchMetricOuptut(
                        key="faithfulness/abpc",
                        value=[desc_r[i][-1] - asc_r[i][-1] for i in range(batch_size)],
                    ),
                ]

    def compute(self) -> dict:
        if not self._results:
            return {}

        def flatten(lst):
            return [x for xs in lst for x in xs]

        accumulated = defaultdict(list)
        for result in self._results:
            for key, values in result.items():
                if isinstance(values[0], list):
                    accumulated[key].extend(flatten(values))
                else:
                    accumulated[key].extend(values)

        mean_results = {}
        for key, value in accumulated.items():
            if key in ["faithfulness/aopc", "faithfulness/abpc"]:
                mean_results[key] = np.mean(value)
            else:
                mean_results[key] = np.mean(value, axis=0)
        return mean_results


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
        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        value = (
            outputs["faithfulness_corr_score"].tolist()
            if not explanation_inputs.is_multi_target
            else [x.tolist() for x in outputs["faithfulness_corr_score"]]
        )
        return [output_cls(key="complexity/faithfulness_corr", value=value)]


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
        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        value = (
            outputs["faithfulness_estimate_score"].tolist()
            if not explanation_inputs.is_multi_target
            else [x.tolist() for x in outputs["faithfulness_estimate_score"]]
        )
        return [output_cls(key="complexity/faithfulness_estimate", value=value)]


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
        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        value = (
            outputs["infidelity_score"].tolist()
            if not explanation_inputs.is_multi_target
            else [x.tolist() for x in outputs["infidelity_score"]]
        )

        return [output_cls(key="complexity/infidelity", value=value)]


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
        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        value = (
            outputs["monotonicity_score"].tolist()
            if not explanation_inputs.is_multi_target
            else [x.tolist() for x in outputs["monotonicity_score"]]
        )

        return [output_cls(key="complexity/monotonicity", value=value)]


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

        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        value = (
            outputs["sensitivity_n_score"].tolist()
            if not explanation_inputs.is_multi_target
            else [x.tolist() for x in outputs["sensitivity_n_score"]]
        )

        return [output_cls(key="complexity/sensitivity_n", value=value)]
