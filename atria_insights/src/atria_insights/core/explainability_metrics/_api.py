"""API functions for loading and preprocessing explainers."""

from __future__ import annotations

from atria_logger import get_logger

from atria_insights.core.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.core.explainability_metrics._registry_group import (
    EXPLAINABILITY_METRIC,
)

logger = get_logger(__name__)


def load_explainability_metric_config(
    explainer_name: str, **kwargs
) -> ExplainabilityMetricConfig:
    return EXPLAINABILITY_METRIC.load_module_config(explainer_name, **kwargs)  # type: ignore
