"""API functions for loading and preprocessing explainers."""

from __future__ import annotations

from atria_logger import get_logger

from atria_insights.perturbation_robustness._registry_groups import (
    perturbation_robustness_PIPELINES,
)
from atria_insights.perturbation_robustness.pipelines._config import (
    PerturbationRobustnessPipelineConfig,
)

logger = get_logger(__name__)


def load_fp_pipeline_config(
    explainer_name: str, **kwargs
) -> PerturbationRobustnessPipelineConfig:
    config = perturbation_robustness_PIPELINES.load_module_config(
        explainer_name, **kwargs
    )
    assert isinstance(config, PerturbationRobustnessPipelineConfig), (
        f"Loaded config is not a FeaturePerturbationPipelineConfig. Found {type(config)=}"
    )
    return config
