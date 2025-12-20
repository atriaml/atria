from __future__ import annotations

import collections
import enum
import os
from typing import TYPE_CHECKING

import numpy as np
from atria_logger import get_logger

if TYPE_CHECKING:
    from typing import Any

    import torch
    from pydantic import BaseModel

logger = get_logger(__name__)

IS_DEBUG_MODE = os.environ.get("ATRIA_LOG_LEVEL", "INFO").upper() == "DEBUG"


class OverflowStrategy(str, enum.Enum):
    """
    Enumeration for overflow strategies.
    """

    select_first = "select_first"
    select_random = "select_random"
    select_all = "select_all"


def _postprocess_qa_predictions(
    words: list[list[str]],
    word_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    question_ids: list[str],
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    n_best_size: int = 20,
    max_answer_length: int = 100,
):
    word_ids = word_ids.tolist()  # type: ignore
    sequence_ids = sequence_ids.tolist()  # type: ignore

    features_per_example = collections.defaultdict(list)
    for feature_id, question_id in enumerate(
        question_ids
    ):  # each example has a unique question id
        features_per_example[question_id].append(feature_id)

    # The dictionaries we have to fill.
    all_predictions_per_question_id = collections.OrderedDict()

    # Let's loop over all the examples!
    for question_id, feature_indices in features_per_example.items():
        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            feature_start_logits = start_logits[feature_index].numpy()
            feature_end_logits = end_logits[feature_index].numpy()

            feature_word_ids = word_ids[feature_index]
            feature_sequence_ids = sequence_ids[feature_index]

            num_question_tokens = 0
            while feature_sequence_ids[num_question_tokens] != 1:
                num_question_tokens += 1

            feature_null_score = feature_start_logits[0] + feature_end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": feature_start_logits[0],
                    "end_logit": feature_end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(feature_start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(feature_end_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index < num_question_tokens
                        or end_index < num_question_tokens
                        or start_index >= len(feature_word_ids)
                        or end_index >= len(feature_word_ids)
                        or feature_word_ids[start_index] is None
                        or feature_word_ids[end_index] is None
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "word_ids": (
                                feature_word_ids[start_index],
                                feature_word_ids[end_index],
                            ),
                            "score": feature_start_logits[start_index]
                            + feature_end_logits[end_index],
                            "start_logit": feature_start_logits[start_index],
                            "end_logit": feature_end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        first_feature_id = features_per_example[question_id][0]
        context = words[first_feature_id]

        for pred in predictions:
            offsets = pred.pop("word_ids")
            pred["text"] = " ".join(
                [x.strip() for x in context[offsets[0] : offsets[1] + 1]]
            )

        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions, strict=True):
            pred["probability"] = prob

        all_predictions_per_question_id[question_ids[feature_index]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, np.float16 | np.float32 | np.float64)
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    return all_predictions_per_question_id


def log_tensor_info(
    data: dict[str, Any] | BaseModel | Any,
    name: str = "tensors",
    max_depth: int = 2,
    current_depth: int = 0,
) -> None:
    import dataclasses
    from dataclasses import is_dataclass

    from pydantic import BaseModel

    indent = "  " * current_depth

    # Convert to dict if needed
    if not IS_DEBUG_MODE:
        return
    if isinstance(data, BaseModel):
        data = data.model_dump()
    elif is_dataclass(data):
        data = {
            field.name: getattr(data, field.name) for field in dataclasses.fields(data)
        }

    if isinstance(data, dict):
        logger.debug(f"{indent}{name} ({len(data)} items):")
        for key, value in data.items():
            _log_single_item(value, f"{indent}  {key}", max_depth, current_depth + 1)
    elif isinstance(data, (list, tuple)):
        logger.debug(f"{indent}{name} ({type(data).__name__} with {len(data)} items):")
        for i, item in enumerate(data):
            _log_single_item(item, f"{indent}  [{i}]", max_depth, current_depth + 1)
    else:
        _log_single_item(data, f"{indent}{name}", max_depth, current_depth)


def _log_single_item(
    item: Any, prefix: str, max_depth: int, current_depth: int
) -> None:
    import torch

    if isinstance(item, torch.Tensor):
        # Tensor information
        dtype_str = str(item.dtype).replace("torch.", "")
        device_str = str(item.device)

        # Memory usage
        memory_mb = item.numel() * item.element_size() / (1024 * 1024)
        memory_str = (
            f"{memory_mb:.2f}MB"
            if memory_mb > 1
            else f"{item.numel() * item.element_size()}B"
        )

        # Statistics for numeric tensors
        stats_str = ""
        if item.dtype.is_floating_point and item.numel() > 0:
            try:
                mean_val = item.float().mean().item()
                std_val = item.float().std().item()
                min_val = item.min().item()
                max_val = item.max().item()
                stats_str = f" | stats: μ={mean_val:.3f} σ={std_val:.3f} range=[{min_val:.3f}, {max_val:.3f}]"
            except:
                stats_str = " | stats: N/A"

        logger.debug(
            f"{prefix}: shape={tuple(item.shape)} | {dtype_str} | {device_str} | {memory_str}{stats_str}"
        )

    elif isinstance(item, (int, float)):
        logger.debug(f"{prefix}: {item} ({type(item).__name__})")

    elif isinstance(item, str):
        preview = item[:50] + "..." if len(item) > 50 else item
        logger.debug(f'{prefix}: "{preview}" (str, len={len(item)})')

    elif isinstance(item, (list, tuple)) and current_depth < max_depth:
        logger.debug(f"{prefix}: {type(item).__name__}({len(item)} items)")
        if len(item) > 0:
            # Show type distribution
            type_counts = {}
            for sub_item in item[:10]:  # Check first 10 items
                item_type = type(sub_item).__name__
                type_counts[item_type] = type_counts.get(item_type, 0) + 1

            type_info = ", ".join(
                [f"{count}×{typ}" for typ, count in type_counts.items()]
            )
            logger.debug(f"{prefix}  └─ types: {type_info}")

            # If all items are tensors, show shape info
            if all(isinstance(x, torch.Tensor) for x in item[:5]):
                shapes = [tuple(x.shape) for x in item[:3]]
                shape_info = ", ".join(str(s) for s in shapes)
                if len(item) > 3:
                    shape_info += ", ..."
                logger.debug(f"{prefix}  └─ tensor shapes: {shape_info}")

    elif isinstance(item, dict) and current_depth < max_depth:
        log_tensor_info(item, prefix.split(": ")[-1], max_depth, current_depth)

    elif item is None:
        logger.debug(f"{prefix}: None")

    else:
        logger.debug(f"{prefix}: {type(item).__name__} | {str(item)[:100]}")


# Enhanced version of your original function
def log_tensors_debug_info(
    tensors: dict[str, torch.Tensor] | BaseModel | Any, title: str = "Tensor Debug Info"
) -> None:
    if not IS_DEBUG_MODE:
        return
    logger.debug(f"╭─ {title}")
    log_tensor_info(tensors, "data")
    logger.debug("╰─" + "─" * (len(title) + 2))
