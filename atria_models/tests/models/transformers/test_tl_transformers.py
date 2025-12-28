from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from atria_models.core.models.transformers._models._lilt._config import (
    LiLTEncoderModelConfig,
)
from atria_models.core.models.transformers._models._lilt._lilt import LiLTEncoderModel


@dataclass
class ModelTestContainer:
    """Simple container for model configs and instances."""

    model_name: str
    tokenizer: Any
    transformers_model: Any
    atria_config: Any
    atria_model: Any
    sample_inputs: dict[str, dict[str, torch.Tensor]]


def inputs_factory(tokenizer: Callable) -> dict[str, dict[str, torch.Tensor]]:
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
    sample_inputs = {
        "single": tokenizer(text=text, boxes=boxes, return_tensors="pt"),
        "batch": tokenizer(
            text=text * 2, boxes=boxes * 2, return_tensors="pt", padding=True
        ),
        "long": tokenizer(text=text * 10, boxes=boxes * 10, return_tensors="pt"),
        "empty": tokenizer(text=[[]], boxes=[[]], return_tensors="pt"),
    }
    return sample_inputs


# Define model configurations
MODEL_CONFIGS = {
    "SCUT-DLVCLab/lilt-roberta-en-base": {
        "atria_config_factory": lambda: LiLTEncoderModelConfig(),
        "atria_model_factory": lambda config: LiLTEncoderModel(config),
        "inputs_factory": inputs_factory,
    }
}


@pytest.fixture(
    scope="session",
    params=["SCUT-DLVCLab/lilt-roberta-en-base"],
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
    container: ModelTestContainer, inputs: dict[str, torch.Tensor]
) -> bool:
    """Compare outputs from both models and return detailed comparison results."""
    container.transformers_model.eval()
    container.atria_model.eval()

    with torch.no_grad():
        hf_inputs = {}
        atria_inputs = {}
        for key in ["input_ids", "attention_mask", "token_type_ids", "bbox"]:
            if key in inputs:
                hf_inputs[key] = inputs[key]

                if key == "input_ids":
                    atria_inputs["token_ids_or_embeddings"] = inputs[key]
                elif key == "attention_mask":
                    atria_inputs["attention_mask"] = inputs[key]
                elif key == "token_type_ids":
                    atria_inputs["token_type_ids_or_embeddings"] = inputs[key]
                elif key == "bbox":
                    atria_inputs["layout_ids_or_embeddings"] = inputs[key]
        hf_output = container.transformers_model(**hf_inputs)
        atria_output = container.atria_model(**atria_inputs)

    values_match = torch.allclose(
        hf_output.last_hidden_state,
        atria_output.last_hidden_state,
        rtol=1e-3,
        atol=1e-5,
    )

    return values_match


@pytest.mark.parametrize("input_type", ["single", "batch", "long"])
def test_model_outputs(model_container: ModelTestContainer, input_type: str):
    inputs = model_container.sample_inputs[input_type]
    values_match = _compare_model_outputs(model_container, inputs)
    assert values_match, f"Model outputs do not match for input type '{input_type}'"
