from atria_metrics.registry.classification import *  # noqa
from atria_metrics.registry.detection import *  # noqa
from atria_metrics.registry.entity_labeling import *  # noqa
from atria_metrics.registry.question_answering import *  # noqa
from atria_metrics.core._registry_group import METRICS

if __name__ == "__main__":
    METRICS.dump(refresh=True)
