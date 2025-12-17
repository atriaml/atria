import torch
from transformers import AutoTokenizer

from atria_models.core.models.transformers.bert.modeling_bert import BertModel
from atria_models.core.models.transformers.bert.modeling_bert_clean_encoder_only import (
    TransformersEncoderModel as BertModelClean,
    TransformersEncoderModelConfig,
)


def load_bert_model() -> BertModel:
    model = BertModel.from_pretrained("bert-base-uncased")
    print("model", model)
    return model


def load_bert_model_clean() -> BertModelClean:
    # model = BertModelClean.from_pretrained("bert-base-uncased")
    model = BertModelClean(TransformersEncoderModelConfig())

    print("model", model)
    return model


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]

model1 = load_bert_model()
model2 = load_bert_model_clean()

named_params1 = model1.named_parameters()
named_params2 = model2.named_parameters()
for (name1, param1), (name2, param2) in zip(named_params1, named_params2, strict=True):
    assert torch.allclose(param1, param2, atol=1e-6), (
        f"Parameters do not match for {name1} and {name2}"
    )
    print(f"Parameters match for {name1} and {name2}")

# model1.eval()
# model2.eval()
# with torch.no_grad():
#     outputs1 = model1(input_ids=input_ids)
# with torch.no_grad():
#     outputs2 = model2(tokens_ids_or_embedding=input_ids)
#     # outputs2 = model2(
#     #     token_embeddings=embeddings.token_embeddings,
#     #     position_embeddings=embeddings.position_embeddings,
#     #     token_type_embeddings=embeddings.token_type_embeddings,
#     # )

# print("Outputs from original BERT model:", outputs1.last_hidden_state)
# print("Outputs from clean BERT model:", outputs2.last_hidden_state)
# assert torch.allclose(
#     outputs1.last_hidden_state, outputs2.last_hidden_state, atol=1e-6
# ), "The outputs of the two BERT models do not match!"
