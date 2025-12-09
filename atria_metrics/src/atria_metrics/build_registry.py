from atria_metrics.detection.cocoeval import *  # noqa
from atria_metrics.instance_classification.ext_modules import *  # noqa
from atria_metrics.instance_classification.f1_score import *  # noqa
from atria_metrics.layout.f1_score import *  # noqa
from atria_metrics.layout.precision import *  # noqa
from atria_metrics.layout.recall import *  # noqa
from atria_metrics.qa.anls import *  # noqa
from atria_metrics.qa.sequence_anls import *  # noqa
from atria_metrics.registry import METRIC
from atria_metrics.token_classification.seqeval import *  # noqa

if __name__ == "__main__":
    METRIC.dump()
