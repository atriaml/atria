from atria_insights.core.explainability_metrics._registry_group import (
    EXPLAINABILITY_METRICS,
)
from atria_insights.core.explainers._registry_group import EXPLAINERS
from atria_insights.core.model_pipelines._image_pipeline import *  # noqa
from atria_insights.core.model_pipelines._registry_groups import (
    EXPLAINABLE_MODEL_PIPELINES,
    # noqa
)
from atria_insights.core.model_pipelines._sequence_pipeline import *  # noqa


def main():
    EXPLAINABLE_MODEL_PIPELINES.dump(refresh=True)
    EXPLAINERS.dump()
    EXPLAINABILITY_METRICS.dump()


if __name__ == "__main__":
    main()
