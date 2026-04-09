from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelOutput:
    loss: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class ClassificationModelOutput(ModelOutput):
    logits: torch.Tensor | None = None
    prediction_probs: torch.Tensor | None = None
    gt_label_value: torch.Tensor | None = None
    gt_label_name: list[str] | None = None
    predicted_label_value: torch.Tensor | None = None
    predicted_label_name: list[str] | None = None

    def is_correct(self) -> bool | None:
        if self.predicted_label_value is not None and self.gt_label_value is not None:
            return (
                (self.predicted_label_value == self.gt_label_value)
                .cpu()
                .numpy()
                .tolist()
            )
        return None


@dataclass(frozen=True)
class TokenClassificationModelOutput(ModelOutput):
    logits: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    predicted_label_names: list[list[str]] | None = None
    target_label_names: list[list[str]] | None = None

    def is_correct(self, token_level: bool = True) -> list[bool] | None:
        if token_level:
            predictions = (
                self.logits.argmax(dim=-1) if self.logits is not None else None
            )
            if predictions is not None and self.token_labels is not None:
                return [
                    [
                        (pred == target).item()
                        for pred, target in zip(pred_seq, target_seq, strict=True)
                    ]
                    for pred_seq, target_seq in zip(
                        predictions, self.token_labels, strict=True
                    )
                ]
        else:
            if (
                self.predicted_label_names is not None
                and self.target_label_names is not None
            ):
                return [
                    [
                        pred == target
                        for pred, target in zip(pred_seq, target_seq, strict=True)
                    ]
                    for pred_seq, target_seq in zip(
                        self.predicted_label_names, self.target_label_names, strict=True
                    )
                ]
            return None


@dataclass(frozen=True)
class LayoutTokenClassificationModelOutput(ModelOutput):
    layout_token_logits: torch.Tensor | None = None
    layout_token_targets: torch.Tensor | None = None
    layout_token_bboxes: torch.Tensor | None = None

    @property
    def layout_token_predictions(self) -> torch.Tensor | None:
        if self.layout_token_logits is not None:
            return torch.argmax(self.layout_token_logits, dim=-1)
        return None

    def is_correct(self) -> list[bool] | None:
        if (
            self.layout_token_predictions is not None
            and self.layout_token_targets is not None
        ):
            return [
                [
                    (pred == target).item()
                    for pred, target in zip(pred_seq, target_seq, strict=True)
                ]
                for pred_seq, target_seq in zip(
                    self.layout_token_predictions,
                    self.layout_token_targets,
                    strict=True,
                )
            ]
        return None


@dataclass(frozen=True)
class QAModelOutput(ModelOutput):
    sample_id: list[str] | None = None
    question: list[str] | None = None
    answer: list[str] | None = None
    gt_answers: list[list[str]] | None = None

    def is_correct(self) -> bool | None:
        from anls import anls_score

        if self.answer is not None and self.gt_answers is not None:
            anls_score = anls_score(
                prediction=self.answer,
                gold_labels=self.gt_answers,  # this takes a list of targets
                threshold=0.5,
            )
        return anls_score > 0.5


@dataclass(frozen=True)
class MMDetEvaluationOutput(ModelOutput):
    loss_dict: dict | None = None
    det_data_samples: list[Any] | None = None
    class_labels: list[str] | None = None
