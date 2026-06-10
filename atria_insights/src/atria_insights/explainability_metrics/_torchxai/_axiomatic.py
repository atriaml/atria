from typing import Any, ClassVar, Literal

import torch
from torchxai.metrics.axiomatic.completeness import completeness
from torchxai.metrics.axiomatic.input_invariance import input_invariance
from torchxai.metrics.axiomatic.monotonicity_corr_and_non_sens import (
    monotonicity_corr_and_non_sens,
)

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainability_metrics._torchxai._base import (
    BatchMetricOuptut,
    ExplainabilityMetric,
    MultiTargetBatchMetricOuptut,
)
from atria_insights.utilities._common import _get_first_layer


@EXPLAINABILITY_METRICS.register("axiomatic/completeness")
class CompletenessConfig(ExplainabilityMetricConfig):
    type: Literal["axiomatic/completeness"] = "axiomatic/completeness"  # type: ignore
    __hash_exclude__: ClassVar[set[str]] = {"enabled"}

    @property
    def name(self):
        return "completeness"


class Completeness(ExplainabilityMetric[CompletenessConfig]):
    __config__ = CompletenessConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> list[BatchMetricOuptut]:
        outputs = completeness(
            forward_func=self._model,
            inputs=explanation_inputs.inputs,
            additional_forward_args=explanation_inputs.additional_forward_args,
            target=self._map_target(explanation_inputs.target),
            attributions=explanations,  # type: ignore
            baselines=explanation_inputs.metric_baselines,
            multi_target=explanation_inputs.is_multi_target,
            return_dict=True,
        )
        assert isinstance(outputs, dict)

        # get the value
        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        value = (
            outputs["score"].tolist()
            if not explanation_inputs.is_multi_target
            else [x.tolist() for x in outputs["score"]]
        )
        return [output_cls(key="axiomatic/completness", value=value)]


@EXPLAINABILITY_METRICS.register("axiomatic/input_invariance")
class InputInvarianceConfig(ExplainabilityMetricConfig):
    type: Literal["axiomatic/input_invariance"] = (  # type: ignore
        "axiomatic/input_invariance"
    )
    constant_shift_value: float = 1.0
    __hash_exclude__: ClassVar[set[str]] = {"enabled"}

    @property
    def name(self):
        return f"input_invariance.csv_{self.constant_shift_value}"


class InputInvariance(ExplainabilityMetric[InputInvarianceConfig]):
    __config__ = InputInvarianceConfig
    _score_keys: ClassVar[list[str]] = ["score"]

    def _input_shifts(
        self, inputs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(inputs, torch.Tensor):
            return torch.ones_like(inputs) * self.config.constant_shift_value
        else:
            return tuple(
                torch.ones_like(inp) * self.config.constant_shift_value
                for inp in inputs
            )

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        # input invariance only works on single input features
        # for each input feature key in the tuple an input layer name is needed
        # this metric is originally designed for single input models such as for image classification
        # models with linear or convolutional first layers
        input_layer_names = [_get_first_layer(self._model)]

        assert isinstance(explanation_inputs.feature_keys, tuple), (
            f"{self.__class__.__name__} only supports single input feature."
            f"Got: {explanation_inputs.feature_keys}"
        )
        assert len(explanation_inputs.feature_keys) == 1, (
            f"{self.__class__.__name__} only supports single input feature."
            f"Got: {explanation_inputs.feature_keys}"
        )

        # input invariance is a explainer based metric
        # it does not use pregenerated explanations but computes them on the go
        outputs = input_invariance(
            explainer=self._explainer,
            # these are additionall explainer forward call args
            # NOTE:
            # notice explainer baselines here
            # this is used to compute attributions on the go during metric computation
            # this metric does not use metric baselines
            inputs=explanation_inputs.inputs,
            target=self._map_target(explanation_inputs.target),
            additional_forward_args=explanation_inputs.additional_forward_args,
            baselines=explanation_inputs.baselines,  # notice explainer baselines, this is different from metric baselines
            feature_mask=explanation_inputs.metric_feature_mask,  # notice explainer feature mask, this is different from metric feature mask
            constant_shifts=self._input_shifts(explanation_inputs.inputs),
            input_layer_names=input_layer_names,
            multi_target=explanation_inputs.is_multi_target,
            return_intermediate_results=False,
            return_dict=True,
        )
        assert isinstance(outputs, dict)

        # get the value
        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        value = (
            outputs["score"].tolist()
            if not explanation_inputs.is_multi_target
            else [x.item() for x in outputs["score"]]
        )
        return [output_cls(key="axiomatic/input_invariance", value=value)]


@EXPLAINABILITY_METRICS.register("axiomatic/monotonicity_corr_and_non_sens")
class MonotonicityCorrAndNonSensConfig(ExplainabilityMetricConfig):
    type: Literal["axiomatic/monotonicity_corr_and_non_sens"] = (  # type: ignore
        "axiomatic/monotonicity_corr_and_non_sens"
    )

    n_perturbations_per_feature: int = 10
    max_features_processed_per_batch: int | None = 40
    percentage_feature_removal_per_step: float = 0
    zero_attribution_threshold: float = 0.00001
    zero_variance_threshold: float = 0.00001
    use_percentage_attribution_threshold: bool = False
    perturb_func: str = "fixed"
    return_intermediate_results: bool = True
    show_progress: bool = True
    return_ratio: bool = True
    __hash_exclude__: ClassVar[set[str]] = {
        "enabled",
        "max_features_processed_per_batch",
        "show_progress",
        "return_intermediate_results",
        "return_ratio",
    }

    @property
    def name(self):
        return (
            f"monotonicity_corr_and_non_sens"
            f".npf_{self.n_perturbations_per_feature}"
            f".pfrs_{self.percentage_feature_removal_per_step}"
            f".zat_{self.zero_attribution_threshold}"
            f".zvt_{self.zero_variance_threshold}"
            f".upat_{int(self.use_percentage_attribution_threshold)}"
            f".pf_{self.perturb_func}"
        )


class MonotonicityCorrAndNonSens(
    ExplainabilityMetric[MonotonicityCorrAndNonSensConfig]
):
    __config__ = MonotonicityCorrAndNonSensConfig
    _score_keys: ClassVar[list[str]] = ["non_sensitivity", "monotonicity_corr"]

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> list[BatchMetricOuptut | MultiTargetBatchMetricOuptut]:
        from torchxai.metrics.axiomatic.monotonicity_corr_and_non_sens import (
            default_fixed_baseline_perturb_func,
        )

        if self.config.perturb_func == "fixed":
            perturb_func = default_fixed_baseline_perturb_func()
        else:
            raise ValueError(
                f"Unsupported perturbation function: {self.config.perturb_func}"
            )
        outputs = monotonicity_corr_and_non_sens(
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
            n_perturbations_per_feature=self.config.n_perturbations_per_feature,
            max_features_processed_per_batch=self.config.max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self.config.percentage_feature_removal_per_step,
            zero_attribution_threshold=self.config.zero_attribution_threshold,
            zero_variance_threshold=self.config.zero_variance_threshold,
            use_percentage_attribution_threshold=self.config.use_percentage_attribution_threshold,
            return_ratio=self.config.return_ratio,
            show_progress=self.config.show_progress,
            return_intermediate_results=self.config.return_intermediate_results,
            multi_target=explanation_inputs.is_multi_target,
            return_dict=True,
        )

        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        return [
            output_cls(
                key="axiomatic/non_sensitivity",
                value=outputs["non_sensitivity"].tolist()
                if not explanation_inputs.is_multi_target
                else [x.tolist() for x in outputs["non_sensitivity"]],
            ),
            output_cls(
                key="faithfulness/monotonicity_corr",
                value=outputs["monotonicity_corr"].tolist()
                if not explanation_inputs.is_multi_target
                else [x.tolist() for x in outputs["monotonicity_corr"]],
            ),
        ]
