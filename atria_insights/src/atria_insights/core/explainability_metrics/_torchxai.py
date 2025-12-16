from torchxai.ignite import BaselineStrategy

from atria_insights.core.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.core.explainability_metrics._registry_group import (
    EXPLAINABILITY_METRIC,
)


@EXPLAINABILITY_METRIC.register("axiomatic/completeness")
class CompletenessMetricConfig(ExplainabilityMetricConfig):
    module_path: str | None = "torchxai.ignite.CompletenessMetric"
    baseline_strategy: BaselineStrategy = BaselineStrategy.zeros
    baselines_fixed_value: float = 0.5
