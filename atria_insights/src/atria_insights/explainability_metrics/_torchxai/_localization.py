from typing import Any, Literal

import torch
from torchxai.metrics import attribution_localization

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainability_metrics._torchxai._base import ExplainabilityMetric


@EXPLAINABILITY_METRICS.register("localization/attr_localization")
class AttrLocalizationConfig(ExplainabilityMetricConfig):
    type: Literal["localization/attr_localization"] = "localization/attr_localization"  # type: ignore
    module_path: str | None = "atria_insights.explainability_metrics.AttrLocalization"
    positive_attributions: bool = True
    weighted: bool = False


class AttrLocalization(ExplainabilityMetric[AttrLocalizationConfig]):
    __config__ = AttrLocalizationConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        featre_mask = self._prepare_feature_mask(explanation_inputs)
        outputs = attribution_localization(
            attributions=explanations,  # type: ignore
            feature_mask=featre_mask,
            multi_target=explanation_inputs.is_multi_target,
            positive_attributions=self.config.positive_attributions,
            weighted=self.config.weighted,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs
