import torch
from transformers import AutoTokenizer
from transformers.models.lilt.modeling_lilt import LiltModel

from atria_models.core.models.transformers._models._lilt import (
    LiLTEncoderModel,
    LiLTEncoderModelConfig,
)

# prepare models
model1 = LiltModel.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
model2 = LiLTEncoderModel(config=LiLTEncoderModelConfig())


# prepare inputs
tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
text = [["hello", "world", "my", "is", "lilt"]]
boxes = [
    [
        [100, 100, 200, 200],
        [150, 150, 250, 250],
        [200, 200, 300, 300],
        [250, 250, 350, 350],
        [300, 300, 400, 400],
    ]
]
inputs = tokenizer(text=text * 10, boxes=boxes * 10, return_tensors="pt")

model1.eval()
model2.eval()
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
bbox = inputs["bbox"]
with torch.no_grad():
    outputs1 = model1(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox)
with torch.no_grad():
    outputs2 = model2(
        token_ids_or_embeddings=input_ids,
        attention_mask=attention_mask,
        layout_ids_or_embeddings=bbox,
    )

print("Outputs from original BERT model:", outputs1.last_hidden_state)
print("Outputs from clean BERT model:", outputs2.last_hidden_state)
assert torch.allclose(
    outputs1.last_hidden_state, outputs2.last_hidden_state, atol=1e-6
), "The outputs of the two BERT models do not match!"
