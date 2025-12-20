from __future__ import annotations

from functools import partial
from typing import Literal

from atria_registry import ModuleConfig

from atria_insights.core.data_types.common import BaselineStrategy


class SimpleBaselineGeneratorConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True
    type: Literal["simple"] = "simple"
    module_path: str | None = "torchxai.ignite._utilities._prepare_baselines_from_type"
    baseline_strategy: BaselineStrategy = BaselineStrategy.zeros
    baselines_fixed_value: float = 0.5

    def build(self) -> partial:  # type: ignore[return]
        from torchxai.ignite._utilities import _prepare_baselines_from_type

        return partial(
            _prepare_baselines_from_type,
            baselines_type=self.baseline_strategy,
            fixed_value=self.baselines_fixed_value,
        )
