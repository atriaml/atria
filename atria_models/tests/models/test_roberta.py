import torch
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel

from atria_models.core.models.transformers._models._roberta import (
    RoBertaEncoderModel,
    RoBertaEncoderModelConfig,
)


def load_bert_model() -> RobertaModel:
    model = RobertaModel.from_pretrained("roberta-base")
    print("model", model)
    return model


def load_bert_model_clean() -> RoBertaEncoderModel:
    config = RoBertaEncoderModelConfig()
    model = RoBertaEncoderModel(config=config)
    return model


tokenizer = AutoTokenizer.from_pretrained("roberta-base")
input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]

model1 = load_bert_model()
model2 = load_bert_model_clean()

# named_params1 = model1.named_parameters()
# named_params2 = model2.named_parameters()
# for (name1, param1), (name2, param2) in zip(named_params1, named_params2, strict=True):
#     print("param1", param1.sum(), param2.sum())
#     assert torch.allclose(param1, param2, atol=1e-6), (
#         f"Parameters do not match for {name1} and {name2}"
#     )
#     print(f"Parameters match for {name1} and {name2}")

model1.eval()
model2.eval()
with torch.no_grad():
    outputs1 = model1(input_ids=input_ids)
with torch.no_grad():
    outputs2 = model2(tokens_ids_or_embedding=input_ids)

print("Outputs from original BERT model:", outputs1.last_hidden_state)
print("Outputs from clean BERT model:", outputs2.last_hidden_state)
assert torch.allclose(
    outputs1.last_hidden_state, outputs2.last_hidden_state, atol=1e-6
), "The outputs of the two BERT models do not match!"
