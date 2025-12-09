from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING

import due_evaluator
from atria_logger import get_logger
from atria_metrics.src.atria_metrics.core.registry_groups import METRIC
from atria_models.core.types.model_outputs import QAModelOutput
from atria_types import DatasetSplitType
from due_evaluator.utils import property_scores_to_string

from atria_metrics.core._base import MetricConfig

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine
    from ignite.metrics.metric import Metric

logger = get_logger(__name__)


_DUE_CONFIG_REPO = "https://github.com/saifullah3396/duetest/blob/main/{dataset_name}/{split}/document.jsonl"


class DueEvalConfig(MetricConfig):
    dataset_name: str
    metric: str
    ignore_case: bool = True

    def build(  # type: ignore
        self,
        device: torch.device | str | None = None,
        num_classes: int | None = None,
        split: DatasetSplitType = DatasetSplitType.test,
    ) -> Metric:
        return super().build(device=device, num_classes=num_classes, split=split)


class DueEvalMetric(Metric):
    def __init__(
        self, dataset: str, split: str, device: str | torch.device = torch.device("cpu")
    ) -> None:
        self._dataset = dataset
        self._split = split
        self._reference_path = self._download_and_cache_reference_file(dataset, split)

        super().__init__(device=device)

    def _download_and_cache_reference_file(self, dataset: str, split: str) -> str:
        import tempfile
        import urllib.request
        from pathlib import Path

        url = _DUE_CONFIG_REPO.format(dataset_name=dataset, split=split)
        cache_dir = Path(tempfile.gettempdir()) / ".atria" / "due_eval"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / f"{dataset}_{split}_reference.jsonl"

        if not local_path.exists():
            logger.info(f"Downloading DUE reference file from {url} to {local_path}")
            urllib.request.urlretrieve(url, str(local_path))
        else:
            logger.info(f"Using cached DUE reference file at {local_path}")

        return str(local_path)

    def reset(self) -> None:
        self._sample_level_qa_pairs = defaultdict(dict)

    def update(self, model_output: QAModelOutput) -> None:
        for qa_pair in model_output.qa_pairs:
            if qa_pair.question not in self._sample_level_qa_pairs[qa_pair.sample_id]:
                self._sample_level_qa_pairs[qa_pair.sample_id][qa_pair.question] = (
                    qa_pair.answer
                )

    def _compute_default(self) -> float:
        reference = []
        answers = []
        logger.info(
            f"Preparing answers and reference for DueEvaluator based on reference file: {self._eval_config.reference_path}"
        )
        with open(self._eval_config.reference_path) as expected:
            for per_sample_reference in expected:
                per_sample_reference = json.loads(per_sample_reference)

                if per_sample_reference["name"] not in self._sample_level_qa_pairs:
                    continue
                predicted_key_values = self._sample_level_qa_pairs[
                    per_sample_reference["name"]
                ]

                per_sample_answers = []
                for ann in per_sample_reference["annotations"]:
                    per_sample_answers.append(
                        {
                            "key": ann["key"],
                            "values": [
                                {"value": predicted_key_values.get(ann["key"], "")}
                            ],  # if no answer found, return empty string
                        }
                    )
                answers.append(
                    {
                        "name": per_sample_reference["name"],
                        "annotations": per_sample_answers,
                    }
                )
                reference.append(per_sample_reference)

        # log info
        assert len(reference) == len(answers), (
            f"Number of samples in reference and answers should be the same. "
            f"Found {len(reference)} in reference and {len(answers)} in answers."
        )
        logger.info("Running DUE Evaluation on %d samples", len(reference))
        logger.info("First reference sample:")
        logger.info(json.dumps(reference[0], indent=2))
        logger.info("First answer sample:")
        logger.info(json.dumps(answers[0], indent=2))

        # load the eval reference file
        evaluator = due_evaluator.DueEvaluator(
            reference=reference,
            answers=answers,
            property_set=None,
            ignore_case=self._eval_config.ignore_case,
            metric=self._eval_config.metric,
        )
        scores = property_scores_to_string(
            [evaluator], "json", ["Precision", "Recall", "F1"]
        )
        scores = json.loads(scores)
        return scores["ALL"]

    def compute(self) -> float:
        if self._eval_config.is_pwc:
            raise NotImplementedError("PWC evaluation not implemented yet.")
        else:
            return self._compute_default()

    def completed(self, engine: Engine, name: str) -> None:
        result = self.compute()
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(
                    f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}"
                )

            for key, value in result.items():
                engine.state.metrics[name + "/" + key] = value
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[name] = result


@METRIC.register("due/DocVQA")
class DocVQAEvalConfig(DueEvalConfig):
    dataset_name: str = "DocVQA"
    metric: str = "ANLS"
    ignore_case: bool = True


@METRIC.register("due/InfographicsVQA")
class InfographicsVQAEvalConfig(DueEvalConfig):
    dataset_name: str = "InfographicsVQA"
    metric: str = "ANLS"
    ignore_case: bool = True


@METRIC.register("due/KleisterCharity")
class KleisterCharityEvalConfig(DueEvalConfig):
    dataset_name: str = "KleisterCharity"
    metric: str = "F1"
    ignore_case: bool = True


@METRIC.register("due/DeepForm")
class DeepFormEvalConfig(DueEvalConfig):
    dataset_name: str = "DeepForm"
    metric: str = "F1"
    ignore_case: bool = True


@METRIC.register("due/WikiTableQuestions")
class WikiTableQuestionsEvalConfig(DueEvalConfig):
    dataset_name: str = "WikiTableQuestions"
    metric: str = "WTQ"
    ignore_case: bool = False


@METRIC.register("due/TabFact")
class TabFactEvalConfig(DueEvalConfig):
    dataset_name: str = "TabFact"
    metric: str = "F1"
    ignore_case: bool = False
