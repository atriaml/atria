import argparse

from atria_insights.explainability_metrics._registry_group import EXPLAINABILITY_METRICS
from atria_insights.explainers._registry_group import EXPLAINERS
from atria_insights.explanation_pipelines._attn_sequence_pipeline import *  # noqa
from atria_insights.explanation_pipelines._image_pipeline import *  # noqa
from atria_insights.explanation_pipelines._registry_groups import (
    EXPLANATION_PIPELINES,
    # noqa
)
from atria_insights.explanation_pipelines._sequence_pipeline import *  # noqa


def main(to_json: bool = False):
    EXPLANATION_PIPELINES.dump(refresh=True, to_json=to_json)
    EXPLAINERS.dump(refresh=True, to_json=to_json)
    EXPLAINABILITY_METRICS.dump(refresh=True, to_json=to_json)

    EXPLANATION_PIPELINES.dump_schema(refresh=True, to_json=to_json)
    EXPLAINERS.dump_schema(refresh=True, to_json=to_json)
    EXPLAINABILITY_METRICS.dump_schema(refresh=True, to_json=to_json)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--to-json", action="store_true", help="Dump schemas to JSON files"
    )
    parsed_args = args.parse_args()
    main(to_json=parsed_args.to_json)
