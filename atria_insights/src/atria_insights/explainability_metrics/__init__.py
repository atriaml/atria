from typing import Annotated

from pydantic import Field

from atria_insights.explainability_metrics._api import load_explainability_metric_config
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainability_metrics._torchxai._axiomatic import (
    CompletenessMetricConfig,
)

ExplainabilityMetricConfigType = Annotated[
    CompletenessMetricConfig, Field(discriminator="type")
]

__all__ = [
    "load_explainability_metric_config",
    "ExplainabilityMetricConfig",
    "EXPLAINABILITY_METRICS",
]
