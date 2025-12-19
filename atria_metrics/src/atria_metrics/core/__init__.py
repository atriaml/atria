# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._base import MetricConfig
    from ._epoch_dict_metric import EpochDictMetric
    from ._output_gatherer import OutputGatherer
    from ._registry_group import METRICS

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_base": ["MetricConfig"],
        "_epoch_dict_metric": ["EpochDictMetric"],
        "_output_gatherer": ["OutputGatherer"],
        "_registry_group": ["METRICS"],
    },
)
