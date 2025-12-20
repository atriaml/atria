from __future__ import annotations

from typing import TYPE_CHECKING

from atria_models.core.types.model_outputs import LayoutTokenClassificationModelOutput

if TYPE_CHECKING:
    import torch


def _get_bsz(input: torch.Tensor | list[torch.Tensor]) -> int:
    return (
        len(input)
        if isinstance(input, list)
        else input.shape[0]
        if input is not None
        else 0
    )


def _flatten(input: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    import torch

    if isinstance(input, list):
        return torch.cat(input)
    elif isinstance(input, torch.Tensor):
        return input.view(-1, *input.shape[2:])
    else:
        raise ValueError("Input is None")


def _output_transform(output: LayoutTokenClassificationModelOutput):
    assert isinstance(output, LayoutTokenClassificationModelOutput), (
        f"Expected {LayoutTokenClassificationModelOutput}, got {type(output)}"
    )
    assert (
        output.layout_token_logits is not None
        and output.layout_token_targets is not None
        and output.layout_token_bboxes is not None
    ), "Logits, targets, and bboxes must not be None"

    logits_bs, target_bs, bbox_bs = (
        _get_bsz(output.layout_token_logits),
        _get_bsz(output.layout_token_targets),
        _get_bsz(output.layout_token_bboxes),
    )
    assert (
        _get_bsz(output.layout_token_logits)
        == _get_bsz(output.layout_token_targets)
        == _get_bsz(output.layout_token_bboxes)
    ), f"Expected same batch size, got {logits_bs}, {target_bs}, {bbox_bs}"

    # flatten across the batch dimension
    flat_logits, flat_targets, flat_bboxes = (
        _flatten(output.layout_token_logits),
        _flatten(output.layout_token_targets),
        _flatten(output.layout_token_bboxes),
    )
    assert flat_logits.shape[0] == flat_targets.shape[0] == flat_bboxes.shape[0], (
        f"Expected same number of samples, got {flat_logits.shape[0]}, {flat_targets.shape[0]}, {flat_bboxes.shape[0]}"
    )

    return flat_logits, flat_targets, flat_bboxes
