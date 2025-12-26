from dataclasses import dataclass

import torch


@dataclass
class AttentionOutput:
    context_output: torch.Tensor
    attentions: torch.Tensor | None = None


@dataclass(frozen=True)
class EncoderLayerOutput:
    hidden_state: torch.Tensor | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass(frozen=True)
class EncoderOutput:
    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass(frozen=True)
class SequenceClassificationHeadOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


@dataclass(frozen=True)
class TokenClassificationHeadOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


@dataclass(frozen=True)
class QuestionAnsweringHeadOutput:
    loss: torch.Tensor | None = None
    start_logits: torch.Tensor | None = None
    end_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class TransformersEncoderModelOutput:
    last_hidden_state: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    head_output: (
        SequenceClassificationHeadOutput
        | TokenClassificationHeadOutput
        | QuestionAnsweringHeadOutput
        | None
    ) = None
