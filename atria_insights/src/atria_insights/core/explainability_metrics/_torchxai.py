from atria_insights.core.data_types.common import BaselineStrategy
from atria_insights.core.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.core.explainability_metrics._registry_group import (
    EXPLAINABILITY_METRICS,
)


@EXPLAINABILITY_METRICS.register("axiomatic/completeness")
class CompletenessMetricConfig(ExplainabilityMetricConfig):
    module_path: str | None = "torchxai.ignite.CompletenessMetric"
    baseline_strategy: BaselineStrategy = BaselineStrategy.zeros
    baselines_fixed_value: float = 0.5
