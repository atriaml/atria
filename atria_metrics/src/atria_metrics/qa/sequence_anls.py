"""
Sequence ANLS Metric Module

This module provides utilities for computing the Average Normalized Levenshtein Similarity (ANLS)
metric for sequence-based question answering tasks. It includes functions for postprocessing
predictions, computing ANLS scores, and defining a sequence ANLS metric.

Functions:
    - convert_to_list: Converts input to a list.
    - postprocess_qa_predictions: Postprocesses predictions for question answering tasks.
    - anls_metric_str: Computes ANLS scores for predictions and gold labels.
    - sequence_anls: Defines a sequence ANLS metric for use in training.

Dependencies:
    - numpy: For numerical operations.
    - textdistance: For computing Levenshtein distance.
    - torch: For PyTorch operations.
    - anls: For ANLS score computation.
    - core.logger: For logging utilities.
    - core.metrics.common.epoch_dict_metric: For defining epoch-level metrics.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable

import textdistance as td
import torch
from atria_core.logger import get_logger
from atria_metrics.common.epoch_dict_metric import EpochDictMetric
from atria_metrics.qa.output_transforms import _sequence_anls_output_transform
from atria_metrics.registry import METRIC
from ignite.metrics import Metric

from anls import anls_score

logger = get_logger(__name__)


def anls_metric_str(
    predictions: list[list[str]], gold_labels: list[list[str]], tau=0.5, rank=0
):
    """
    Computes ANLS scores for predictions and gold labels.

    Args:
        predictions: List of predicted answers.
        gold_labels: List of gold answers, each instance may have multiple gold labels.
        tau: Threshold for normalized Levenshtein similarity.
        rank: Rank of the computation.

    Returns:
        A tuple containing:
        - res: List of ANLS scores for each instance.
        - Average ANLS score across all instances.
    """
    res = []
    for _, (preds, golds) in enumerate(zip(predictions, gold_labels, strict=True)):
        max_s = 0
        for pred in preds:
            for gold in golds:
                dis = td.levenshtein.distance(pred.lower(), gold.lower())
                max_len = max(len(pred), len(gold))
                if max_len == 0:
                    s = 0
                else:
                    nl = dis / max_len
                    s = 1 - nl if nl < tau else 0
                max_s = max(s, max_s)
        res.append(max_s)
    return res, sum(res) / len(res)


@METRIC.register("sequence_anls", output_transform=_sequence_anls_output_transform)
def sequence_anls(
    output_transform: Callable, device: str | torch.device, threshold: float = 0.5
) -> Metric:
    """
    Defines a sequence ANLS metric for use in training.

    Args:
        output_transform: Function to transform the output.
        device: Device to perform computations on.
        threshold: Threshold for ANLS score computation.

    Returns:
        An instance of EpochDictMetric for computing sequence ANLS.
    """

    def wrap(predicted_answers: list[str], gold_answers: list[list[str]]):
        assert len(predicted_answers) == len(gold_answers), (
            f"The number of predicted answers must match the lists of ground truth answers."
            f"len(predicted_answers){len(predicted_answers)} != len(gold_answers)({len(gold_answers)})"
        )
        logger.info("Computing sequence ANLS...")
        logger.info(f"Total predictions:, {len(predicted_answers)}")
        logger.info(f"Total ground truths: {len(gold_answers)}")
        logger.info(f"Predicted answers batch: {predicted_answers[:20]}")
        logger.info(f"Target answers: {gold_answers[:20]}")
        anls_scores = [
            anls_score(pred, target, threshold=threshold)
            for pred, target in zip(predicted_answers, gold_answers, strict=True)
        ]
        anls = sum(anls_scores) / len(anls_scores)
        return anls

    return EpochDictMetric(wrap, output_transform=output_transform, device=device)
