"""API functions for loading and preprocessing explainers."""

from __future__ import annotations

from atria_logger import get_logger
from pydantic import TypeAdapter

from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS

logger = get_logger(__name__)


def load_explainability_metric_config(
    explainer_name: str, **kwargs
) -> ExplainabilityMetricConfig:
    return EXPLAINABILITY_METRICS.load_module_config(explainer_name, **kwargs)  # type: ignore


def load_explainer_config(
    explainer_name: str, **kwargs
) -> ExplainabilityMetricConfigType:
    config = EXPLAINABILITY_METRICS.load_module_config(explainer_name, **kwargs)
    adapter = TypeAdapter(ExplainabilityMetricConfigType)
    return adapter.validate_python(config)
