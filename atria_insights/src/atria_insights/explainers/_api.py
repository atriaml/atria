"""API functions for loading and preprocessing explainers."""

from __future__ import annotations

from atria_logger import get_logger
from pydantic import TypeAdapter

from atria_insights.explainers._registry_group import EXPLAINERS
from atria_insights.explainers._torchxai import ExplainerConfigType

logger = get_logger(__name__)


def load_explainer_config(explainer_name: str, **kwargs) -> ExplainerConfigType:
    config = EXPLAINERS.load_module_config(explainer_name, **kwargs)
    adapter = TypeAdapter(ExplainerConfigType)
    return adapter.validate_python(config)
