from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from atria_models.core.models.transformers._models._layoutlmv3._config import (
    LayoutLMv3EncoderModelConfig,
)
from atria_models.core.models.transformers._models._layoutlmv3._layoutlmv3 import (
    LayoutLMv3EncoderModel,
)


@dataclass
class ModelTestContainer:
    """Simple container for model configs and instances."""

    model_name: str
    tokenizer: Any
    transformers_model: Any
    atria_config: Any
    atria_model: Any
    sample_inputs: dict[str, dict[str, torch.Tensor]]


def layoutlmv3_input_factory(tokenizer: Callable) -> dict[str, dict[str, torch.Tensor]]:
    from datasets import load_dataset

    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
    example = dataset[0]
    image = example["image"]
    words = example["tokens"]
    boxes = example["bboxes"]
    word_labels = example["ner_tags"]

    # Single example
    single_encoding = processor(
        image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt"
    )
    single_encoding.pop("labels", None)

    # Batch example (duplicate the single example to form a batch)
    batch_images = [image, image]
    batch_words = [words, words]
    batch_boxes = [boxes, boxes]
    batch_word_labels = [word_labels, word_labels]
    batch_encoding = processor(
        batch_images,
        batch_words,
        boxes=batch_boxes,
        word_labels=batch_word_labels,
        return_tensors="pt",
    )
    batch_encoding.pop("labels", None)

    sample_inputs = {"single": single_encoding, "batch": batch_encoding}
    return sample_inputs


# Define model configurations
MODEL_CONFIGS = {
    "microsoft/layoutlmv3-base": {
        "atria_config_factory": lambda: LayoutLMv3EncoderModelConfig(),
        "atria_model_factory": lambda config: LayoutLMv3EncoderModel(config),
        "inputs_factory": layoutlmv3_input_factory,
    }
}


@pytest.fixture(
    scope="session",
    params=["microsoft/layoutlmv3-base"],
    ids=lambda x: x.replace("-", "_"),
)
def model_container(request) -> ModelTestContainer:
    """Create a model test container for the specified model."""
    model_name = request.param
    config = MODEL_CONFIGS[model_name]

    # Load tokenizer and transformers model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers_model = AutoModel.from_pretrained(model_name)

    # Create atria config and model
    atria_config = config["atria_config_factory"]()
    atria_model = config["atria_model_factory"](atria_config)
    sample_inputs = config["inputs_factory"](tokenizer)

    return ModelTestContainer(
        model_name=model_name,
        tokenizer=tokenizer,
        transformers_model=transformers_model,
        atria_config=atria_config,
        atria_model=atria_model,
        sample_inputs=sample_inputs,
    )


def _compare_model_outputs(
    container: ModelTestContainer,
    inputs: dict[str, torch.Tensor],
    add_bbox: bool = False,
    add_image: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Compare outputs from both models and return detailed comparison results."""
    container.transformers_model.eval()
    container.atria_model.eval()

    with torch.no_grad():
        hf_inputs = {}
        atria_inputs = {}

        hf_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        atria_inputs = {
            "token_ids_or_embeddings": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if add_bbox:
            hf_inputs["bbox"] = inputs["bbox"]
            atria_inputs["layout_ids_or_embeddings"] = inputs["bbox"]
            atria_inputs["layout_ids"] = inputs["bbox"]

        if add_image:
            hf_inputs["pixel_values"] = inputs["pixel_values"]
            atria_inputs["image"] = inputs["pixel_values"]
        hf_output = container.transformers_model(**hf_inputs)
        atria_output = container.atria_model(**atria_inputs)

    # the model outputs should be close but cannot be exactly equal due to the layer_norm not giving
    # deterministic results
    values_match = torch.allclose(
        hf_output.last_hidden_state,
        atria_output.last_hidden_state,
        atol=atol,
        rtol=rtol,
    )

    return values_match


@pytest.mark.parametrize("input_type", ["single", "batch"])
def test_model_outputs(model_container: ModelTestContainer, input_type: str):
    inputs = model_container.sample_inputs[input_type]
    values_match = _compare_model_outputs(model_container, inputs)
    values_match = _compare_model_outputs(model_container, inputs, add_bbox=True)
    values_match = _compare_model_outputs(
        model_container, inputs, add_bbox=True, add_image=True
    )
    assert values_match, f"Model outputs do not match for input type '{input_type}'"
