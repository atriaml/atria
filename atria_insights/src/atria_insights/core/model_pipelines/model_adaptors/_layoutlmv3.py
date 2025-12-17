from __future__ import annotations

from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification,
)

from atria_insights.core.model_pipelines.model_adaptors._sequence import (
    DocumentExplainabilityAdaptor,
)


class LayoutLMv3ModelAdaptor(DocumentExplainabilityAdaptor):
    def __init__(
        self,
        model: LayoutLMv3ForSequenceClassification
        | LayoutLMv3ForTokenClassification
        | LayoutLMv3ForQuestionAnswering,
    ) -> None:
        super().__init__(model=model)
        self._model = model
