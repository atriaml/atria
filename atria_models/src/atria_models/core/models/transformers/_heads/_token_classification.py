import torch
from atria_logger import get_logger
from torch import nn
from torch.nn.functional import cross_entropy

from atria_models.core.models.transformers._configs._encoder_model import (
    ClassificationSubTask,
)
from atria_models.core.models.transformers._outputs import TokenClassificationHeadOutput

logger = get_logger(__name__)


class TokenClassificationHead(nn.Module):
    def __init__(
        self,
        num_labels: int,
        subtask: ClassificationSubTask = ClassificationSubTask.single_label_classification,
        classifier_dropout: float | None = None,
        hidden_size: int | None = None,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.sub_task = subtask
        self.classifier_dropout = classifier_dropout
        self.hidden_size = hidden_size
        self._build_layers()

    def _build_layers(self):
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def _get_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

    def forward(
        self, last_hidden_state: torch.Tensor, labels: torch.Tensor | None = None
    ) -> TokenClassificationHeadOutput:
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        loss = self._get_loss(logits, labels) if labels is not None else None
        return TokenClassificationHeadOutput(loss=loss, logits=logits)
