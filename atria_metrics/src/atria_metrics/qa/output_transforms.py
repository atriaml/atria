from __future__ import annotations

from typing import TYPE_CHECKING

from atria_core.types import QAModelOutput, SequenceQAModelOutput

if TYPE_CHECKING:
    import torch


def _anls_output_transform(output: QAModelOutput) -> tuple[list[str], list[str]]:
    assert isinstance(output, QAModelOutput), (
        f"Expected {QAModelOutput}, got {type(output)}"
    )
    return output.pred_answers, output.target_answers


def _sequence_anls_output_transform(
    output: SequenceQAModelOutput,
) -> tuple[list[str], list[int], list[int], str, torch.Tensor, torch.Tensor, list[str]]:
    assert isinstance(output, SequenceQAModelOutput), (
        f"Expected {SequenceQAModelOutput}, got {type(output)}"
    )
    assert len(output.predicted_answers) == len(output.gold_answers), (
        "Length of predicted_answers and gold_answers must be the same"
        ", got "
        f"output.predicted_answers={len(output.predicted_answers)} and "
        f"output.gold_answers={len(output.gold_answers)}"
    )
    return [
        prediction[0]["text"] for prediction in output.predicted_answers
    ], output.gold_answers
