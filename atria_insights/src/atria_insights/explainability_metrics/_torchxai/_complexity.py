from typing import Any, Literal

import torch
from torchxai.metrics.complexity.complexity_entropy import (
    complexity_entropy,
    complexity_entropy_feature_grouped,
)
from torchxai.metrics.complexity.complexity_sundararajan import (
    complexity_sundararajan,
    complexity_sundararajan_feature_grouped,
)
from torchxai.metrics.complexity.effective_complexity import effective_complexity
from torchxai.metrics.complexity.sparseness import (
    sparseness,
    sparseness_feature_grouped,
)

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainability_metrics._torchxai._base import ExplainabilityMetric


@EXPLAINABILITY_METRICS.register("complexity/complexity_entropy")
class ComplexityEntropyConfig(ExplainabilityMetricConfig):
    type: Literal["complexity/complexity_entropy"] = "complexity/complexity_entropy"  # type: ignore
    module_path: str | None = "atria_insights.explainability_metrics.ComplexityEntropy"
    group_features: bool = False


class ComplexityEntropy(ExplainabilityMetric[ComplexityEntropyConfig]):
    __config__ = ComplexityEntropyConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        if self.config.group_features:
            outputs = complexity_entropy_feature_grouped(
                attributions=explanations,
                feature_mask=self._prepare_feature_mask(explanation_inputs),
                multi_target=explanation_inputs.is_multi_target,
                return_dict=True,
            )
        else:
            outputs = complexity_entropy(
                attributions=explanations,
                multi_target=explanation_inputs.is_multi_target,
                return_dict=True,
            )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("complexity/complexity_s")
class ComplexitySConfig(ExplainabilityMetricConfig):
    type: Literal["complexity/complexity_s"] = "complexity/complexity_s"  # type: ignore
    module_path: str | None = "atria_insights.explainability_metrics.ComplexityS"
    group_features: bool = False
    eps: float = 0.00001
    normalize_attribution: bool = True


class ComplexityS(ExplainabilityMetric[ComplexitySConfig]):
    __config__ = ComplexitySConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        if self.config.group_features:
            outputs = complexity_sundararajan_feature_grouped(
                attributions=explanations,
                feature_mask=self._prepare_feature_mask(explanation_inputs),
                eps=self.config.eps,
                normalize_attribution=self.config.normalize_attribution,
                multi_target=explanation_inputs.is_multi_target,
                return_dict=True,
            )
        else:
            outputs = complexity_sundararajan(
                attributions=explanations,
                eps=self.config.eps,
                normalize_attribution=self.config.normalize_attribution,
                multi_target=explanation_inputs.is_multi_target,
                return_dict=True,
            )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("complexity/sparseness")
class SparsenessConfig(ExplainabilityMetricConfig):
    type: Literal["complexity/sparseness"] = "complexity/sparseness"  # type: ignore
    module_path: str | None = "atria_insights.explainability_metrics.Sparseness"
    group_features: bool = False


class Sparseness(ExplainabilityMetric[SparsenessConfig]):
    __config__ = SparsenessConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        if self.config.group_features:
            outputs = sparseness_feature_grouped(
                attributions=explanations,
                feature_mask=self._prepare_feature_mask(explanation_inputs),
                multi_target=explanation_inputs.is_multi_target,
                return_dict=True,
            )
        else:
            outputs = sparseness(
                attributions=explanations,
                multi_target=explanation_inputs.is_multi_target,
                return_dict=True,
            )
        assert isinstance(outputs, dict)
        return outputs


@EXPLAINABILITY_METRICS.register("complexity/effective_complexity")
class EffectiveComplexityConfig(ExplainabilityMetricConfig):
    type: Literal["complexity/effective_complexity"] = (  # type: ignore
        "complexity/effective_complexity"
    )
    module_path: str | None = (
        "atria_insights.explainability_metrics.EffectiveComplexity"
    )

    n_perturbations_per_feature: int = 10
    max_features_processed_per_batch: int | None = None
    percentage_feature_removal_per_step: float = 0
    zero_variance_threshold: float = 0.01
    use_percentage_attribution_threshold: bool = False
    perturb_func: str = "fixed"
    return_intermediate_results: bool = True
    show_progress: bool = False
    return_ratio: bool = False


class EffectiveComplexity(ExplainabilityMetric[EffectiveComplexityConfig]):
    __config__ = EffectiveComplexityConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        from torchxai.metrics.complexity.effective_complexity import (
            default_fixed_baseline_perturb_func,
        )

        if self.config.perturb_func == "fixed":
            perturb_func = default_fixed_baseline_perturb_func
        else:
            raise ValueError(
                f"Unsupported perturbation function: {self.config.perturb_func}"
            )

        return effective_complexity(
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
            n_perturbations_per_feature=self.config.n_perturbations_per_feature,
            max_features_processed_per_batch=self.config.max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self.config.percentage_feature_removal_per_step,
            zero_variance_threshold=self.config.zero_variance_threshold,
            return_ratio=self.config.return_ratio,
            show_progress=self.config.show_progress,
            return_intermediate_results=self.config.return_intermediate_results,
            multi_target=explanation_inputs.is_multi_target,
            return_dict=True,
        )
