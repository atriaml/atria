from torch import Tensor
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import

class BertModel:
    def __init__(self) -> None:
        self._model = AutoModel.from_pretrained("bert-base-uncased")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
