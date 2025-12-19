from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import due_evaluator
from atria_logger import get_logger
from atria_models.core.types.model_outputs import QAModelOutput
from due_evaluator.utils import property_scores_to_string
from ignite.metrics.metric import Metric

from atria_metrics.core._base import MetricConfig

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine

logger = get_logger(__name__)


_DUE_CONFIG_REPO = "https://github.com/saifullah3396/duetest/blob/main/{dataset_name}/{split}/document.jsonl"


class DueEvalMetric(Metric):
    def __init__(
        self,
        dataset: str,
        split: str,
        device: str | torch.device = "cpu",
        metric: str = "F1",
        ignore_case: bool = True,
    ) -> None:
        import torch

        self._dataset = dataset
        self._split = split
        self._metric = metric
        self._ignore_case = ignore_case
        self._reference_path = self._download_and_cache_reference_file(dataset, split)

        super().__init__(device=torch.device(device))

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

    def update(self, output: QAModelOutput) -> None:
        assert output.qa_pairs is not None, (
            "model_output.qa_pairs is None. Cannot update DueEvalMetric."
        )
        for qa_pair in output.qa_pairs:
            if qa_pair.question not in self._sample_level_qa_pairs[qa_pair.sample_id]:
                self._sample_level_qa_pairs[qa_pair.sample_id][qa_pair.question] = (
                    qa_pair.answer
                )

    def compute(self) -> float:
        reference = []
        answers = []
        logger.info(
            f"Preparing answers and reference for DueEvaluator based on reference file: {self._reference_path}"
        )
        with open(self._reference_path) as expected:
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
            ignore_case=self._ignore_case,
            metric=self._metric,
        )
        scores = property_scores_to_string(
            [evaluator], "json", ["Precision", "Recall", "F1"]
        )
        scores = json.loads(scores)
        return scores["ALL"]

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


class DueEvalMetricConfig(MetricConfig):
    module_path: str | None = (
        "atria_metrics.core.question_answering.due_eval.DueEvalMetric"
    )
    dataset_name: str
    metric: str
    ignore_case: bool = True

    def build(  # type: ignore
        self,
        stage: Literal["validation", "test"],
        device: torch.device | str | None = None,
    ) -> Metric:
        if stage == "validation":
            split = "dev"
        elif stage == "test":
            split = "test"
        else:
            raise ValueError(
                f"DueEvalMetricConfig can only be built for validation or test stages. Found: {stage}"
            )
        return DueEvalMetric(
            dataset=self.dataset_name,
            split=split,
            device=device if device is not None else torch.device("cpu"),
            metric=self.metric,
            ignore_case=self.ignore_case,
        )
