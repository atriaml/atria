from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from atria_models.core.models.transformers._models._bert import (
    BertEncoderModel,
    BertEncoderModelConfig,
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


# Define model configurations
MODEL_CONFIGS = {
    "bert-base-uncased": {
        "atria_config_factory": lambda: BertEncoderModelConfig(),
        "atria_model_factory": lambda config: BertEncoderModel(config),
    }
}


@pytest.fixture(
    scope="session", params=["bert-base-uncased"], ids=lambda x: x.replace("-", "_")
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

    # Create sample inputs
    sample_inputs = {
        "single": tokenizer("Hello, my dog is cute", return_tensors="pt"),
        "batch": tokenizer(
            ["Hello world", "How are you?"], return_tensors="pt", padding=True
        ),
        "long": tokenizer(
            "This is a longer sentence that contains multiple words to test model behavior",
            return_tensors="pt",
        ),
        "empty": tokenizer("", return_tensors="pt"),
    }

    return ModelTestContainer(
        model_name=model_name,
        tokenizer=tokenizer,
        transformers_model=transformers_model,
        atria_config=atria_config,
        atria_model=atria_model,
        sample_inputs=sample_inputs,
    )


def compare_layer_params(
    layer1: nn.Module,
    layer2: nn.Module,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    verbose: bool = False,
) -> bool:
    params1 = dict(layer1.named_parameters())
    params2 = dict(layer2.named_parameters())

    if params1.keys() != params2.keys():
        if verbose:
            missing_1 = params2.keys() - params1.keys()
            missing_2 = params1.keys() - params2.keys()
            print("Parameter name mismatch.")
            print(f"Only in layer2: {missing_1}")
            print(f"Only in layer1: {missing_2}")
        return False

    for name in params1:
        p1 = params1[name]
        p2 = params2[name]

        if p1.shape != p2.shape:
            if verbose:
                print(f"Shape mismatch for '{name}': {p1.shape} vs {p2.shape}")
            return False

        if not torch.allclose(p1, p2, rtol=rtol, atol=atol):
            if verbose:
                max_diff = (p1 - p2).abs().max().item()
                print(f"Parameter '{name}' differs (max diff = {max_diff})")
            return False

    return True


def _compare_embeddings(
    container: ModelTestContainer, inputs: dict[str, torch.Tensor]
) -> bool:
    """Compare outputs from both models and return detailed comparison results."""
    container.transformers_model.eval()
    container.atria_model.eval()

    with torch.no_grad():

        def get_bert_embeddings(
            input_ids, token_type_ids, position_ids=None, past_key_values_length=0
        ):
            input_shape = input_ids.size()
            seq_length = input_shape[1]

            if position_ids is None:
                position_ids = container.transformers_model.embeddings.position_ids[
                    :, past_key_values_length : seq_length + past_key_values_length
                ]

            inputs_embeds = container.transformers_model.embeddings.word_embeddings(
                input_ids
            )
            token_type_embeddings = (
                container.transformers_model.embeddings.token_type_embeddings(
                    token_type_ids
                )
            )
            position_embeddings = (
                container.transformers_model.embeddings.position_embeddings(
                    position_ids
                )
            )
            return inputs_embeds + token_type_embeddings + position_embeddings

        embedding_output = get_bert_embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=None,
            past_key_values_length=0,
        )
        atria_output = container.atria_model.ids_to_embeddings(
            token_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=None,
        )
        atria_embedding_output = atria_output.sum()
        assert torch.allclose(embedding_output, atria_embedding_output), (
            "Embeddings do not match before layer norm"
        )
        compare_layer_params(
            container.transformers_model.embeddings.LayerNorm,
            container.atria_model.embeddings_aggregator.layer_norm,
            verbose=True,
        )

        # apply layer norm
        embedding_output = container.transformers_model.embeddings.LayerNorm(
            embedding_output
        )
        atria_embedding_output = container.transformers_model.embeddings.LayerNorm(
            atria_embedding_output
        )
        assert torch.allclose(embedding_output, atria_embedding_output, atol=1.0e-6), (
            "Embeddings after layer norm do not match"
        )


def _compare_model_outputs(
    container: ModelTestContainer, inputs: dict[str, torch.Tensor]
) -> bool:
    """Compare outputs from both models and return detailed comparison results."""
    container.transformers_model.eval()
    container.atria_model.eval()

    with torch.no_grad():
        hf_output = container.transformers_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        atria_output = container.atria_model(
            tokens_ids_or_embedding=inputs["input_ids"],
            token_type_ids_or_embeddings=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
        )

    values_match = torch.allclose(
        hf_output.last_hidden_state,
        atria_output.last_hidden_state,
        rtol=1e-3,
        atol=1e-5,
    )

    return values_match


@pytest.mark.parametrize("input_type", ["single", "batch", "long", "empty"])
def test_model_outputs(model_container: ModelTestContainer, input_type: str):
    inputs = model_container.sample_inputs[input_type]
    values_match = _compare_model_outputs(model_container, inputs)
    assert values_match, f"Model outputs do not match for input type '{input_type}'"
