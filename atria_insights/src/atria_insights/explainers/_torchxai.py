from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from atria_insights.explainers._base import ExplainerConfig
from atria_insights.explainers._registry_group import EXPLAINERS

if TYPE_CHECKING:
    import torch
    from torchxai.explainers import Explainer


class GradExplainerConfig(ExplainerConfig):
    def build(  # type: ignore
        self,
        model: torch.nn.Module,
        internal_batch_size: int = 1,
        multi_target: bool = False,
        grad_batch_size: int = 1,
        **kwargs,
    ) -> Explainer:
        return super().build(
            model=model,
            internal_batch_size=internal_batch_size,
            grad_batch_size=grad_batch_size,
            multi_target=multi_target,
            **kwargs,
        )


@EXPLAINERS.register("grad/saliency")
class SaliencyExplainerConfig(ExplainerConfig):
    type: Literal["grad/saliency"] = "grad/saliency"
    module_path: str | None = "torchxai.explainers.SaliencyExplainer"


@EXPLAINERS.register("grad/integrated_gradients")
class IntegratedGradientsExplainerConfig(ExplainerConfig):
    type: Literal["grad/integrated_gradients"] = "grad/integrated_gradients"
    module_path: str | None = "torchxai.explainers.IntegratedGradientsExplainer"
    n_steps: int = 50


@EXPLAINERS.register("grad/deeplift")
class DeepLiftExplainerConfig(ExplainerConfig):
    type: Literal["grad/deeplift"] = "grad/deeplift"
    module_path: str | None = "torchxai.explainers.DeepLiftExplainer"


@EXPLAINERS.register("grad/deeplift_shap")
class DeepLiftShapExplainerConfig(ExplainerConfig):
    type: Literal["grad/deeplift_shap"] = "grad/deeplift_shap"
    module_path: str | None = "torchxai.explainers.DeepLiftShapExplainer"


@EXPLAINERS.register("grad/gradient_shap")
class GradientShapExplainerConfig(ExplainerConfig):
    type: Literal["grad/gradient_shap"] = "grad/gradient_shap"
    module_path: str | None = "torchxai.explainers.GradientShapExplainer"
    n_samples: int = 25


@EXPLAINERS.register("grad/guided_backprop")
class GuidedBackpropExplainerConfig(ExplainerConfig):
    type: Literal["grad/guided_backprop"] = "grad/guided_backprop"
    module_path: str | None = "torchxai.explainers.GuidedBackpropExplainer"


@EXPLAINERS.register("grad/input_x_gradient")
class InputXGradientExplainerConfig(ExplainerConfig):
    type: Literal["grad/input_x_gradient"] = "grad/input_x_gradient"
    module_path: str | None = "torchxai.explainers.InputXGradientExplainer"


@EXPLAINERS.register("perturbation/feature_ablation")
class FeatureAblationExplainerConfig(ExplainerConfig):
    type: Literal["perturbation/feature_ablation"] = "perturbation/feature_ablation"
    module_path: str | None = "torchxai.explainers.FeatureAblationExplainer"
    weight_attributions: bool = True


@EXPLAINERS.register("perturbation/kernel_shap")
class KernelShapExplainerConfig(ExplainerConfig):
    type: Literal["perturbation/kernel_shap"] = "perturbation/kernel_shap"
    module_path: str | None = "torchxai.explainers.KernelShapExplainer"
    n_samples: int = 25
    weight_attributions: bool = True


@EXPLAINERS.register("perturbation/lime")
class LimeExplainerConfig(ExplainerConfig):
    type: Literal["perturbation/lime"] = "perturbation/lime"
    module_path: str | None = "torchxai.explainers.LimeExplainer"
    n_samples: int = 25
    alpha: float = 0.01
    weight_attributions: bool = True


@EXPLAINERS.register("perturbation/occlusion")
class OcclusionExplainerConfig(ExplainerConfig):
    type: Literal["perturbation/occlusion"] = "perturbation/occlusion"
    module_path: str | None = "torchxai.explainers.OcclusionExplainer"
    sliding_window_shapes: tuple[int, ...] | tuple[tuple[int, ...], ...] = (3, 16, 16)
    strides: None | int | tuple[int, ...] | tuple[int | tuple[int, ...], ...] = (
        3,
        4,
        4,
    )


@EXPLAINERS.register("random")
class RandomExplainerConfig(ExplainerConfig):
    type: Literal["random"] = "random"
    module_path: str | None = "torchxai.explainers.RandomExplainer"


ExplainerConfigType = Annotated[
    SaliencyExplainerConfig
    | IntegratedGradientsExplainerConfig
    | DeepLiftExplainerConfig
    | DeepLiftShapExplainerConfig
    | GradientShapExplainerConfig
    | GuidedBackpropExplainerConfig
    | InputXGradientExplainerConfig
    | FeatureAblationExplainerConfig
    | KernelShapExplainerConfig
    | LimeExplainerConfig
    | OcclusionExplainerConfig
    | RandomExplainerConfig,
    Field(discriminator="type"),
]
