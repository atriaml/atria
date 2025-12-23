from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from atria_registry import ModuleConfig

from atria_insights.baseline_generators._base import BaselineGenerator
from atria_insights.data_types._common import BaselineStrategy


class SimpleBaselineGeneratorConfig(ModuleConfig):
    module_path: str | None = (
        "atria_insights.baseline_generators._simple.SimpleBaselineGenerator"
    )
    type: Literal["simple"] = "simple"
    baseline_strategy: BaselineStrategy = BaselineStrategy.zeros
    baselines_fixed_value: float = 0.0


class SimpleBaselineGenerator(BaselineGenerator[SimpleBaselineGeneratorConfig]):
    __config__ = SimpleBaselineGeneratorConfig

    def __call__(  # type: ignore[override]
        self, inputs: torch.Tensor | OrderedDict[str, torch.Tensor]
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        from torchxai.ignite._utilities import _prepare_baselines_from_type

        if isinstance(inputs, OrderedDict):
            baselines = OrderedDict()
            for key, tensor in inputs.items():
                baselines[key] = _prepare_baselines_from_type(
                    tensor,
                    baselines_type=self.config.baseline_strategy,  # type: ignore[arg-type]
                    fixed_value=self.config.baselines_fixed_value,
                )
            return baselines
        else:
            return _prepare_baselines_from_type(
                inputs,
                baselines_type=self.config.baseline_strategy,  # type: ignore[arg-type]
                fixed_value=self.config.baselines_fixed_value,
            )
