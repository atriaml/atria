"""API functions for loading and preprocessing explainers."""

from __future__ import annotations

from atria_logger import get_logger

from atria_insights.explainers._base import ExplainerConfig

logger = get_logger(__name__)


def load_explainer_config(explainer_name: str, **kwargs) -> ExplainerConfig:
    from atria_insights.explainers._registry_group import EXPLAINER

    return EXPLAINER.load_module_config(explainer_name, **kwargs)  # type: ignore
