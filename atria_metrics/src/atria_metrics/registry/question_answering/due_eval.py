from __future__ import annotations

from atria_logger import get_logger

from atria_metrics.core._registry_group import METRICS
from atria_metrics.core.question_answering.due_eval import DueEvalMetricConfig

logger = get_logger(__name__)


@METRICS.register("due/DocVQA")
class DocVQAEvalConfig(DueEvalMetricConfig):
    dataset_name: str = "DocVQA"
    metric: str = "ANLS"
    ignore_case: bool = True


@METRICS.register("due/InfographicsVQA")
class InfographicsVQAEvalConfig(DueEvalMetricConfig):
    dataset_name: str = "InfographicsVQA"
    metric: str = "ANLS"
    ignore_case: bool = True


@METRICS.register("due/KleisterCharity")
class KleisterCharityEvalConfig(DueEvalMetricConfig):
    dataset_name: str = "KleisterCharity"
    metric: str = "F1"
    ignore_case: bool = True


@METRICS.register("due/DeepForm")
class DeepFormEvalConfig(DueEvalMetricConfig):
    dataset_name: str = "DeepForm"
    metric: str = "F1"
    ignore_case: bool = True


@METRICS.register("due/WikiTableQuestions")
class WikiTableQuestionsEvalConfig(DueEvalMetricConfig):
    dataset_name: str = "WikiTableQuestions"
    metric: str = "WTQ"
    ignore_case: bool = False


@METRICS.register("due/TabFact")
class TabFactEvalConfig(DueEvalMetricConfig):
    dataset_name: str = "TabFact"
    metric: str = "F1"
    ignore_case: bool = False
