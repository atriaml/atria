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
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


@dataclass(frozen=True)
class TokenClassificationHeadOutput:
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


@dataclass(frozen=True)
class QuestionAnsweringHeadOutput:
    loss: torch.FloatTensor | None = None
    start_logits: torch.FloatTensor | None = None
    end_logits: torch.FloatTensor | None = None


@dataclass(frozen=True)
class TransformersEncoderModelOutput:
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass(frozen=True)
class SequenceClassificationModelOutput(TransformersEncoderModelOutput):
    head_output: SequenceClassificationHeadOutput | None = None


@dataclass(frozen=True)
class TokenClassificationModelOutput(TransformersEncoderModelOutput):
    head_output: TokenClassificationHeadOutput | None = None


@dataclass(frozen=True)
class QuestionAnsweringModelOutput(TransformersEncoderModelOutput):
    head_output: QuestionAnsweringHeadOutput | None = None
