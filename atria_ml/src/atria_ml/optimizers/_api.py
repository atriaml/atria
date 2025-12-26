"""API functions for loading and preprocessing optimizers."""

from __future__ import annotations

from atria_logger import get_logger
from pydantic.type_adapter import TypeAdapter

from atria_ml.optimizers._configs import OptimizerConfigType

logger = get_logger(__name__)


def load_optimizer_config(optimizer_name: str, **kwargs) -> OptimizerConfigType:
    from atria_ml.optimizers._registry_group import OPTIMIZERS

    config = OPTIMIZERS.load_module_config(optimizer_name, **kwargs)  # type: ignore
    adapter = TypeAdapter(OptimizerConfigType)
    return adapter.validate_python(config)
