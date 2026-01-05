from typing import Any, Literal

import h5py
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
    mask_h5_file_path: str | None = None


class AttrLocalization(ExplainabilityMetric[AttrLocalizationConfig]):
    __config__ = AttrLocalizationConfig

    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        with h5py.File(self.config.mask_h5_file_path, "r") as f:
            feature_mask = {
                feature_key: [] for feature_key in explanation_inputs.feature_keys
            }
            for feature_key in explanation_inputs.feature_keys:
                for sample_id in explanation_inputs.sample_id:
                    feature_mask[feature_key].append(
                        torch.from_numpy(f[sample_id][feature_key][:])
                    )
                feature_mask[feature_key] = torch.cat(feature_mask[feature_key]).to(
                    explanations[0][0].device
                )
        outputs = attribution_localization(
            attributions=explanations,  # type: ignore
            # this is actually wrong, but we don't use this metric. If needed, we need to load feature localization masks from a file
            feature_mask=feature_mask,
            multi_target=explanation_inputs.is_multi_target,
            positive_attributions=self.config.positive_attributions,
            weighted=self.config.weighted,
            return_dict=True,
        )
        assert isinstance(outputs, dict)
        return outputs
