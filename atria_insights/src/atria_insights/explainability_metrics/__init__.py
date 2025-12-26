from typing import Annotated

from pydantic import Field

from atria_insights.explainability_metrics._api import load_explainability_metric_config
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainability_metrics._torchxai._axiomatic import (
    Completeness,
    CompletenessConfig,
    InputInvariance,
    InputInvarianceConfig,
    MonotonicityCorrAndNonSens,
    MonotonicityCorrAndNonSensConfig,
)
from atria_insights.explainability_metrics._torchxai._base import ExplainabilityMetric
from atria_insights.explainability_metrics._torchxai._complexity import (
    ComplexityEntropy,
    ComplexityEntropyConfig,
    ComplexityS,
    ComplexitySConfig,
    EffectiveComplexity,
    EffectiveComplexityConfig,
    Sparseness,
    SparsenessConfig,
)
from atria_insights.explainability_metrics._torchxai._faithfulness import (
    AOPC,
    AOPCConfig,
    FaithfulnessCorrelation,
    FaithfulnessCorrelationConfig,
    FaithfulnessEstimate,
    FaithfulnessEstimateConfig,
    Infidelity,
    InfidelityConfig,
    Monotonicity,
    MonotonicityConfig,
    SensitivityN,
    SensitivityNConfig,
)
from atria_insights.explainability_metrics._torchxai._localization import (
    AttrLocalization,
    AttrLocalizationConfig,
)
from atria_insights.explainability_metrics._torchxai._robustness import (
    SensitivityMaxAvg,
    SensitivityMaxAvgConfig,
)

ExplainabilityMetricConfigType = Annotated[
    ExplainabilityMetricConfig, Field(discriminator="type")
]

__all__ = [
    "load_explainability_metric_config",
    "ExplainabilityMetricConfig",
    "ExplainabilityMetric",
    "EXPLAINABILITY_METRICS",
    # axiomatic metrics
    "CompletenessConfig",
    "Completeness",
    "InputInvarianceConfig",
    "InputInvariance",
    "MonotonicityCorrAndNonSens",
    "MonotonicityCorrAndNonSensConfig",
    # complexity metrics
    "ComplexityEntropyConfig",
    "ComplexityEntropy",
    "ComplexitySConfig",
    "ComplexityS",
    "EffectiveComplexityConfig",
    "EffectiveComplexity",
    "SparsenessConfig",
    "Sparseness",
    # faithfulness metrics
    "AOPCConfig",
    "AOPC",
    "FaithfulnessCorrelationConfig",
    "FaithfulnessCorrelation",
    "FaithfulnessEstimateConfig",
    "FaithfulnessEstimate",
    "InfidelityConfig",
    "Infidelity",
    "Infidelity",
    "InfidelityConfig",
    "SensitivityN",
    "SensitivityNConfig",
    "Monotonicity",
    "MonotonicityConfig",
    # robustness metrics
    "SensitivityMaxAvg",
    "SensitivityMaxAvgConfig",
    # localization metrics
    "AttrLocalization",
    "AttrLocalizationConfig",
]
