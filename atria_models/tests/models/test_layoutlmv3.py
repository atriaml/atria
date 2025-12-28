import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from transformers.models.layoutlmv3 import LayoutLMv3Model

from atria_models.core.models.transformers._models._layoutlmv3 import (
    LayoutLMv3EncoderModel,
    LayoutLMv3EncoderModelConfig,
)

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
word_labels = example["ner_tags"]

encoding = processor(
    image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt"
)
encoding.pop("labels")

# prepare models
model1 = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
model2 = LayoutLMv3EncoderModel(config=LayoutLMv3EncoderModelConfig())
model1.eval()
model2.eval()
device = "cpu"


def _set_all_random_seeds(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Configure CuDNN backend for deterministic behavior if required
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)


def test1(add_bbox: bool = False, add_image: bool = False):
    inputs1 = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }
    inputs2 = {
        "token_ids_or_embeddings": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }
    if add_bbox:
        inputs1["bbox"] = encoding["bbox"].to(device)
        inputs2["layout_ids_or_embeddings"] = encoding["bbox"].to(device)
        inputs2["layout_ids"] = encoding["bbox"].to(device)
    if add_image:
        inputs1["pixel_values"] = encoding["pixel_values"].to(device)
        inputs2["image"] = encoding["pixel_values"].to(device)
    _set_all_random_seeds(1234)
    outputs1 = model1(**inputs1)
    _set_all_random_seeds(1234)
    outputs2 = model2(**inputs2)
    print("outputs1", outputs1)
    print("outputs2", outputs2)
    assert torch.allclose(
        outputs1.last_hidden_state, outputs2.last_hidden_state, atol=1e-6
    ), (
        f"Outputs are not close enough: {outputs1.last_hidden_state} vs {outputs2.last_hidden_state}, {outputs1.last_hidden_state - outputs2.last_hidden_state}"
    )


with torch.no_grad():
    test1()
    test1(add_bbox=True)
    test1(add_bbox=True, add_image=True)
