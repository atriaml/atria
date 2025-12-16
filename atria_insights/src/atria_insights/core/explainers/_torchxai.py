from atria_insights.src.atria_insights.explainers._base import ExplainerConfig
from atria_insights.src.atria_insights.registry import EXPLAINER


@EXPLAINER.register("grad/saliency")
class SaliencyExplainerConfig(ExplainerConfig):
    module_path: str | None = "torchxai.explainers._grad.saliency.SaliencyExplainer"


@EXPLAINER.register("grad/integrated_gradients")
class IntegratedGradientsExplainerConfig(ExplainerConfig):
    module_path: str | None = (
        "torchxai.explainers._grad.integrated_gradients.IntegratedGradientsExplainer"
    )
    n_steps: int = 50


@EXPLAINER.register("grad/deeplift")
class DeepLiftExplainerConfig(ExplainerConfig):
    module_path: str | None = "torchxai.explainers._grad.deeplift.DeepLiftExplainer"


@EXPLAINER.register("grad/deeplift_shap")
class DeepLiftShapExplainerConfig(ExplainerConfig):
    module_path: str | None = (
        "torchxai.explainers._grad.deeplift_shap.DeepLiftShapExplainer"
    )


@EXPLAINER.register("grad/gradient_shap")
class GradientShapExplainerConfig(ExplainerConfig):
    module_path: str | None = (
        "torchxai.explainers._grad.gradient_shap.GradientShapExplainer"
    )
    internal_batch_size: int = 1
    n_samples: int = 25


@EXPLAINER.register("grad/guided_backprop")
class GuidedBackpropExplainerConfig(ExplainerConfig):
    module_path: str | None = (
        "torchxai.explainers._grad.guided_backprop.GuidedBackpropExplainer"
    )


@EXPLAINER.register("grad/input_x_gradient")
class InputXGradientExplainerConfig(ExplainerConfig):
    module_path: str | None = (
        "torchxai.explainers._grad.input_x_gradient.InputXGradientExplainer"
    )


@EXPLAINER.register("perturbation/feature_ablation")
class FeatureAblationExplainerConfig(ExplainerConfig):
    module_path: str | None = (
        "torchxai.explainers._perturbation.feature_ablation.FeatureAblationExplainer"
    )
    weight_attributions: bool = True


@EXPLAINER.register("perturbation/kernel_shap")
class KernelShapExplainerConfig(ExplainerConfig):
    module_path: str | None = (
        "torchxai.explainers._perturbation.kernel_shap.KernelShapExplainer"
    )
    internal_batch_size: int = 1
    n_samples: int = 25
    weight_attributions: bool = True


@EXPLAINER.register("perturbation/lime")
class LimeExplainerConfig(ExplainerConfig):
    module_path: str | None = "torchxai.explainers._perturbation.lime.LimeExplainer"
    internal_batch_size: int = 1
    n_samples: int = 25
    alpha: float = 0.01
    weight_attributions: bool = True


@EXPLAINER.register("perturbation/occlusion")
class OcclusionExplainerConfig(ExplainerConfig):
    module_path: str | None = (
        "torchxai.explainers._perturbation.occlusion.OcclusionExplainer"
    )
    sliding_window_shapes: tuple[int, ...] | tuple[tuple[int, ...], ...] = (3, 16, 16)
    strides: None | int | tuple[int, ...] | tuple[int | tuple[int, ...], ...] = (
        3,
        4,
        4,
    )
    internal_batch_size: int = 1


@EXPLAINER.register("random")
class RandomExplainerConfig(ExplainerConfig):
    module_path: str | None = "torchxai.explainers.random.RandomExplainer"
