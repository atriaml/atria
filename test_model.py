"""Atria Model Builder Base Class Module"""

from typing import Any

from atria_models.core.models import Model, ModelConfig


class TestModelConfig(ModelConfig):
    x: int = 10
    y: float = 5.0


class TestModel(Model[TestModelConfig]):
    __config__ = TestModelConfig

    def forward(self, input_tensor: Any) -> Any:
        print(f"Input Tensor: {input_tensor}")
        print(f"Config x: {self.config.x}, Config y: {self.config.y}")


model = TestModel(x=10, y=3.14)

print(model)
model(input_tensor="Sample Input")
