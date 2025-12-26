from typing import Any, Literal

import torch
from torchxai.metrics import sensitivity_max_and_avg

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainability_metrics._torchxai._base import ExplainabilityMetric


@EXPLAINABILITY_METRICS.register("robustness/sensitivity_max_and_avg")
class SensitivityMaxAvgConfig(ExplainabilityMetricConfig):
    type: Literal["robustness/sensitivity_max_and_avg"] = (  # type: ignore
        "robustness/sensitivity_max_and_avg"
    )
    module_path: str | None = "atria_insights.explainability_metrics.SensitivityMaxAvg"
    perturb_radius: float = 0.02
    n_perturb_samples: int = 10
    norm_ord: str = "fro"
    max_examples_per_batch: int | None = None


class SensitivityMaxAvg(ExplainabilityMetric[SensitivityMaxAvgConfig]):
    __config__ = SensitivityMaxAvgConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        outputs = sensitivity_max_and_avg(
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
            feature_mask=explanation_inputs.feature_mask,  # notice explainer feature mask, this is different from metric feature mask
            perturb_radius=self.config.perturb_radius,
            n_perturb_samples=self.config.n_perturb_samples,
            norm_ord=self.config.norm_ord,
            max_examples_per_batch=self.config.max_examples_per_batch,
            multi_target=explanation_inputs.is_multi_target,
            return_intermediate_results=False,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs
