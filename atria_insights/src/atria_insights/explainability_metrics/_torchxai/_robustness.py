from typing import Any, ClassVar, Literal

import torch
from torchxai.metrics import sensitivity_max_and_avg

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainability_metrics._torchxai._base import (
    BatchMetricOuptut,
    ExplainabilityMetric,
    MultiTargetBatchMetricOuptut,
)


@EXPLAINABILITY_METRICS.register("robustness/sensitivity_max_and_avg")
class SensitivityMaxAvgConfig(ExplainabilityMetricConfig):
    type: Literal["robustness/sensitivity_max_and_avg"] = (  # type: ignore
        "robustness/sensitivity_max_and_avg"
    )
    perturb_radius: float = 0.02
    n_perturb_samples: int = 10
    norm_ord: str = "fro"
    max_examples_per_batch: int | None = None
    __hash_exclude__: ClassVar[set[str]] = {"enabled", "max_examples_per_batch"}

    @property
    def name(self):
        return (
            f"sensitivity_max_and_avg"
            f".pr_{self.perturb_radius}"
            f".nps_{self.n_perturb_samples}"
            f".no_{self.norm_ord}"
        )


class SensitivityMaxAvg(ExplainabilityMetric[SensitivityMaxAvgConfig]):
    __config__ = SensitivityMaxAvgConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        if explanation_inputs.is_multi_target:
            self._explainer.multi_target = True

            # multi target currently only supports max_examples_per_batch = 1
            # this is because we we have multiple targets per sample where each target
            # is not expanded for repeated inputs in batch
            max_examples_per_batch = 1
        else:
            max_examples_per_batch = self.config.max_examples_per_batch
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
            feature_mask=explanation_inputs.metric_feature_mask,  # notice explainer feature mask, this is different from metric feature mask
            sliding_window_shapes=explanation_inputs.sliding_window_shapes,  # needed for occlusion
            strides=explanation_inputs.strides,  # needed for occlusion
            perturb_radius=self.config.perturb_radius,
            n_perturb_samples=self.config.n_perturb_samples,
            norm_ord=self.config.norm_ord,
            max_examples_per_batch=max_examples_per_batch,
            multi_target=explanation_inputs.is_multi_target,
            attention_token_target=explanation_inputs.attention_token_target,  # needed for attention explainers with token targets
            feature_keys=explanation_inputs.feature_keys,  # needed for feature mask segmentors
            return_intermediate_results=False,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        output_cls = (
            BatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetBatchMetricOuptut
        )
        return [
            output_cls(
                key="axiomatic/sensitivity_max",
                value=outputs["sensitivity_max"].tolist()
                if not explanation_inputs.is_multi_target
                else [x.tolist() for x in outputs["sensitivity_max"]],
            ),
            output_cls(
                key="faithfulness/sensitivity_avg",
                value=outputs["sensitivity_avg"].tolist()
                if not explanation_inputs.is_multi_target
                else [x.tolist() for x in outputs["sensitivity_avg"]],
            ),
        ]
