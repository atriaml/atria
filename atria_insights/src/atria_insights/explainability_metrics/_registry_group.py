from __future__ import annotations

from typing import TYPE_CHECKING

from atria_registry import ModuleRegistry, RegistryGroup

if TYPE_CHECKING:
    from atria_insights.explainability_metrics._base import (
        ExplainabilityMetricConfig,  # noqa
    )


class ExplainabilityMetricsRegistryGroup(RegistryGroup["ExplainabilityMetricConfig"]):
    """Registry group for explainers."""

    pass


ModuleRegistry().add_registry_group(
    name="EXPLAINABILITY_METRIC",
    registry_group=ExplainabilityMetricsRegistryGroup(
        name="explainability_metric", package="atria_insights"
    ),
)
EXPLAINABILITY_METRICS: ExplainabilityMetricsRegistryGroup = (
    ModuleRegistry().get_registry_group("EXPLAINABILITY_METRIC")
)  # type: ignore
