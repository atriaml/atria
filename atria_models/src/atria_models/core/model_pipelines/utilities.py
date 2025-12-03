from __future__ import annotations

import collections
import enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


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
