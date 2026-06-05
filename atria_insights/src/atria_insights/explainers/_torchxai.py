from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from atria_registry._module_base import ModuleConfig
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
        **kwargs,
    ) -> Explainer:
        return ModuleConfig.build(
            self,
            model=model,
            internal_batch_size=internal_batch_size,
            multi_target=multi_target,
            **kwargs,
        )


@EXPLAINERS.register("grad/saliency")
class SaliencyExplainerConfig(GradExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.SaliencyExplainer"
    type: Literal["grad/saliency"] = "grad/saliency"


@EXPLAINERS.register("grad/integrated_gradients")
class IntegratedGradientsExplainerConfig(GradExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.IntegratedGradientsExplainer"
    type: Literal["grad/integrated_gradients"] = "grad/integrated_gradients"
    n_steps: int = 50


@EXPLAINERS.register("grad/deeplift")
class DeepLiftExplainerConfig(GradExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.DeepLiftExplainer"
    type: Literal["grad/deeplift"] = "grad/deeplift"


@EXPLAINERS.register("grad/deeplift_shap")
class DeepLiftShapExplainerConfig(GradExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.DeepLiftShapExplainer"
    type: Literal["grad/deeplift_shap"] = "grad/deeplift_shap"


@EXPLAINERS.register("grad/gradient_shap")
class GradientShapExplainerConfig(GradExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.GradientShapExplainer"
    type: Literal["grad/gradient_shap"] = "grad/gradient_shap"
    n_samples: int = 25


@EXPLAINERS.register("grad/guided_backprop")
class GuidedBackpropExplainerConfig(GradExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.GuidedBackpropExplainer"
    type: Literal["grad/guided_backprop"] = "grad/guided_backprop"


@EXPLAINERS.register("grad/input_x_gradient")
class InputXGradientExplainerConfig(GradExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.InputXGradientExplainer"
    type: Literal["grad/input_x_gradient"] = "grad/input_x_gradient"


@EXPLAINERS.register("perturbation/feature_ablation")
class FeatureAblationExplainerConfig(ExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.FeatureAblationExplainer"
    type: Literal["perturbation/feature_ablation"] = "perturbation/feature_ablation"
    weight_attributions: bool = True


@EXPLAINERS.register("perturbation/kernel_shap")
class KernelShapExplainerConfig(ExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.KernelShapExplainer"
    type: Literal["perturbation/kernel_shap"] = "perturbation/kernel_shap"
    n_samples: int = 25
    weight_attributions: bool = True


@EXPLAINERS.register("perturbation/lime")
class LimeExplainerConfig(ExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.LimeExplainer"
    type: Literal["perturbation/lime"] = "perturbation/lime"
    n_samples: int = 25
    alpha: float = 0.01
    weight_attributions: bool = True


@EXPLAINERS.register("perturbation/occlusion")
class OcclusionExplainerConfig(ExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.OcclusionExplainer"
    type: Literal["perturbation/occlusion"] = "perturbation/occlusion"


@EXPLAINERS.register("random")
class RandomExplainerConfig(ExplainerConfig):
    __module_path__: ClassVar[str] = "torchxai.explainers.RandomExplainer"
    type: Literal["random"] = "random"


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
