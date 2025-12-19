# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Ensure registry is initialized immediately
import atria_metrics.core._registry_group  # noqa: F401

if TYPE_CHECKING:
    from atria_metrics.core._base import MetricConfig
    from atria_metrics.core._epoch_dict_metric import EpochDictMetric
    from atria_metrics.core._output_gatherer import OutputGatherer
    from atria_metrics.core._registry_group import METRICS

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "core._base": ["MetricConfig"],
        "core._epoch_dict_metric": ["EpochDictMetric"],
        "core._output_gatherer": ["OutputGatherer"],
        "core._registry_group": ["METRICS"],
    },
)
