import torch
from atria_logger import get_logger
from torch import nn
from torch.nn.functional import cross_entropy

from atria_models.core.models.transformers._outputs import QuestionAnsweringHeadOutput

logger = get_logger(__name__)


class QAHead(nn.Module):
    def __init__(self, classifier_dropout: float = 0.1, hidden_size: int = 768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        classifier_dropout = classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class QuestionAnsweringHead(nn.Module):
    def __init__(self, hidden_size: int = 768, classifier_dropout: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout
        self._build_layers()

    def _build_layers(self):
        self.qa_outputs = QAHead(
            hidden_size=self.hidden_size, classifier_dropout=self.classifier_dropout
        )

    def _get_loss(
        self,
        start_positions: torch.Tensor,
        start_logits: torch.Tensor,
        end_positions: torch.Tensor,
        end_logits: torch.Tensor,
    ) -> torch.Tensor:
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)
        start_loss = cross_entropy(
            start_logits, start_positions, ignore_index=ignored_index
        )
        end_loss = cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
    ) -> QuestionAnsweringHeadOutput:
        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self._get_loss(
                start_positions, start_logits, end_positions, end_logits
            )

        return QuestionAnsweringHeadOutput(
            loss=loss, start_logits=start_logits, end_logits=end_logits
        )
