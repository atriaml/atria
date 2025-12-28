import torch
from atria_logger import get_logger
from torch import nn
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cross_entropy,
    mse_loss,
)

from atria_models.core.models.transformers._configs._encoder_model import (
    ClassificationSubTask,
)
from atria_models.core.models.transformers._modules._pooler import DefaultPooler
from atria_models.core.models.transformers._outputs import (
    SequenceClassificationHeadOutput,
)

logger = get_logger(__name__)


class SequenceClassificationHead(nn.Module):
    def __init__(
        self,
        num_labels: int,
        subtask: ClassificationSubTask = ClassificationSubTask.single_label_classification,
        classifier_dropout: float = 0.1,
        hidden_size: int = 768,
        add_pooling_layer: bool = True,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.sub_task = subtask
        self.classifier_dropout = classifier_dropout
        self.hidden_size = hidden_size
        self.add_pooling_layer = add_pooling_layer

        self._build_layers()

    def _build_layers(self):
        self.pooler = (
            DefaultPooler(self.hidden_size)
            if self.add_pooling_layer and self.hidden_size is not None
            else None
        )
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def _get_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        match self.sub_task:
            case ClassificationSubTask.regression:
                if self.num_labels == 1:
                    return mse_loss(logits.squeeze(), labels.squeeze())
                return mse_loss(logits, labels)

            case ClassificationSubTask.single_label_classification:
                return cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

            case ClassificationSubTask.multi_label_classification:
                return binary_cross_entropy_with_logits(logits, labels)

            case _:
                raise ValueError(f"Unknown sub_task {self.sub_task}")

    def forward(
        self, last_hidden_state: torch.Tensor, labels: torch.Tensor | None = None
    ) -> SequenceClassificationHeadOutput:
        last_hidden_state = (
            self.pooler(last_hidden_state)
            if self.pooler is not None
            else last_hidden_state[:, 0]
        )
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        loss = self._get_loss(logits, labels) if labels is not None else None
        return SequenceClassificationHeadOutput(loss=loss, logits=logits)
