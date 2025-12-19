from abc import ABC, abstractmethod

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from atria_models.core.models.transformers._models._bert import (
    BertEncoderModel,
    BertEncoderModelConfig,
)


class BaseTransformerTestCase(ABC):
    """Base test class for transformer models."""

    @abstractmethod
    def get_atria_model(self):
        """Return the atria model instance."""
        pass

    @abstractmethod
    def get_atria_config(self):
        """Return the atria model config."""
        pass


@pytest.mark.parametrize("model_name", ["bert-base-uncased", "distilbert-base-uncased"])
class TestTransformerModels(BaseTransformerTestCase):
    @pytest.fixture(scope="class", autouse=True)
    def setup_models(self, request, model_name):
        """Setup tokenizer and models for the test class."""
        request.cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
        request.cls.transformers_model = AutoModel.from_pretrained(model_name)
        request.cls.atria_model = self.get_atria_model()

        # Setup sample inputs
        request.cls.sample_inputs = {
            "single": request.cls.tokenizer(
                "Hello, my dog is cute", return_tensors="pt"
            )["input_ids"],
            "batch": request.cls.tokenizer(
                ["Hello world", "How are you?"], return_tensors="pt", padding=True
            )["input_ids"],
            "long": request.cls.tokenizer(
                "This is a longer sentence that contains multiple words to test model behavior",
                return_tensors="pt",
            )["input_ids"],
            "empty": request.cls.tokenizer("", return_tensors="pt")["input_ids"],
        }

    def get_atria_model(self):
        """Return BERT model - override for other models."""
        return BertEncoderModel(self.get_atria_config())

    def get_atria_config(self):
        """Return BERT config - override for other models."""
        return BertEncoderModelConfig()

    def _get_model_outputs(self, input_ids):
        """Get outputs from both models."""
        self.transformers_model.eval()
        self.atria_model.eval()

        with torch.no_grad():
            hf_output = self.transformers_model(input_ids=input_ids)
            atria_output = self.atria_model(tokens_ids_or_embedding=input_ids)

        return hf_output, atria_output

    @pytest.mark.parametrize("input_type", ["single", "batch", "long"])
    def test_output_shapes_match(self, input_type):
        """Test that both models produce outputs with the same shape."""
        input_ids = self.sample_inputs[input_type]
        hf_output, atria_output = self._get_model_outputs(input_ids)

        assert hf_output.last_hidden_state.shape == atria_output.last_hidden_state.shape

    def test_output_values_match(self):
        """Test that both models produce similar output values."""
        input_ids = self.sample_inputs["single"]
        hf_output, atria_output = self._get_model_outputs(input_ids)

        assert torch.allclose(
            hf_output.last_hidden_state, atria_output.last_hidden_state, atol=1e-6
        ), "Model outputs do not match!"

    def test_deterministic_output(self):
        """Test that the atria model produces deterministic outputs."""
        input_ids = self.sample_inputs["single"]

        self.atria_model.eval()
        with torch.no_grad():
            output1 = self.atria_model(tokens_ids_or_embedding=input_ids)
            output2 = self.atria_model(tokens_ids_or_embedding=input_ids)

        assert torch.allclose(output1.last_hidden_state, output2.last_hidden_state)

    def test_config_initialization(self):
        """Test that model initializes with custom config."""
        config = self.get_atria_config()
        model = self.get_atria_model()

        assert model is not None
        assert hasattr(model, "config")

    @pytest.mark.parametrize("training_mode", [True, False])
    def test_training_modes(self, training_mode):
        """Test that models can be set to train/eval mode."""
        if training_mode:
            self.transformers_model.train()
            self.atria_model.train()
        else:
            self.transformers_model.eval()
            self.atria_model.eval()

        assert self.transformers_model.training == training_mode
        assert self.atria_model.training == training_mode

    def test_empty_input_handling(self):
        """Test model behavior with empty input."""
        input_ids = self.sample_inputs["empty"]

        self.atria_model.eval()
        with torch.no_grad():
            output = self.atria_model(tokens_ids_or_embedding=input_ids)

        assert output.last_hidden_state is not None
        assert output.last_hidden_state.dim() == 3

    @pytest.mark.parametrize("input_type", ["single", "batch", "long"])
    def test_sequence_length_consistency(self, input_type):
        """Test model output dimensions match input dimensions."""
        input_ids = self.sample_inputs[input_type]

        self.atria_model.eval()
        with torch.no_grad():
            output = self.atria_model(tokens_ids_or_embedding=input_ids)

        assert output.last_hidden_state.shape[0] == input_ids.shape[0]  # batch size
        assert (
            output.last_hidden_state.shape[1] == input_ids.shape[1]
        )  # sequence length


# Example of how to create tests for specific models
class TestBertModels(TestTransformerModels):
    """BERT-specific tests."""

    pass


class TestDistilBertModels(TestTransformerModels):
    """DistilBERT-specific tests - would need different atria model."""

    def get_atria_model(self):
        # Return DistilBERT atria model when available
        return BertEncoderModel(self.get_atria_config())  # placeholder
