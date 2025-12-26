import fire
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_insights.baseline_generators import (
    FeatureBasedBaselineGeneratorConfig,
    SimpleBaselineGeneratorConfig,
)
from atria_insights.data_types._common import BaselineStrategy
from atria_insights.explainability_metrics._api import load_explainability_metric_config
from atria_insights.explainers._api import load_explainer_config
from atria_insights.explainers._torchxai import (  # noqa
    DeepLiftExplainerConfig,
    DeepLiftShapExplainerConfig,
    SaliencyExplainerConfig,
)
from atria_insights.feature_segmentors import GridSegmenterConfig
from atria_ml.configs._task import TrainingTaskConfig

from atria_insights import ExplanationTaskConfig, ModelExplainer
from atria_logger import get_logger

logger = get_logger(__name__)

_EXPLAINERS = [
    "grad/saliency",
    "grad/integrated_gradients",
    "grad/deeplift",
    "grad/deeplift_shap",
    "grad/gradient_shap",
    "grad/guided_backprop",
    "grad/input_x_gradient",
    "perturbation/feature_ablation",
    "perturbation/kernel_shap",
    "perturbation/lime",
    "perturbation/occlusion",
    "random",
]

_METRICS = [
    "axiomatic/completeness",
    "axiomatic/monotonicity_corr_and_non_sens",
    "complexity/complexity_entropy",
    "complexity/complexity_s",
    "complexity/effective_complexity",
    "complexity/sparseness",
    "faithfulness/aopc",
    "faithfulness/faithfulness_correlation",
    "faithfulness/faithfulness_estimate",
    "faithfulness/infidelity",
    "faithfulness/monotonicity",
    "faithfulness/sensitivity_n",
    "robustness/sensitivity_max_and_avg",
]

_DEFAULT_BASELINES_GENERATOR_CONFIG = SimpleBaselineGeneratorConfig(
    baseline_strategy=BaselineStrategy.fixed,
    baselines_fixed_value=0.0,  # This corresponds to mean after normalization
)
_DEFAULT_DEEPSHAP_BASELINES_GENERATOR_CONFIG = FeatureBasedBaselineGeneratorConfig(
    num_baselines=10
)

_DEFAULT_FEATURE_SEGMENTOR_CONFIG = GridSegmenterConfig(cell_size=16)


def _load_config_from_checkpoint(checkpoint_path: str) -> TrainingTaskConfig:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert "config" in checkpoint, "No config found in checkpoint."
    return TrainingTaskConfig.from_dict(checkpoint["config"])


def main(
    checkpoint_path: str,
    dataset_name: str | None = None,
    explainer_name: str = "grad/saliency",
    exp_name: str = "explain_img_cls_00",
    output_dir: str = "./outputs",
    batch_size: int = 32,
):
    assert explainer_name in _EXPLAINERS, f"Explainer {explainer_name} not recognized."

    baselines_generator = _DEFAULT_BASELINES_GENERATOR_CONFIG
    if explainer_name in ["grad/deeplift_shap"]:
        baselines_generator = _DEFAULT_DEEPSHAP_BASELINES_GENERATOR_CONFIG

    explainability_metrics = None
    if len(_METRICS) > 0:
        explainability_metrics = {
            key: load_explainability_metric_config(key) for key in _METRICS
        }
        logger.debug(f"Loaded explainability metrics: {explainability_metrics.keys()}")

    explanation_task_config = ExplanationTaskConfig.from_training_task_config(
        training_task_config=_load_config_from_checkpoint(checkpoint_path),
        dataset_name=dataset_name,
        exp_name=exp_name,
        output_dir=output_dir,
        explainer=load_explainer_config(explainer_name),
        baseline_generator=baselines_generator,
        feature_segmentor=_DEFAULT_FEATURE_SEGMENTOR_CONFIG,
        internal_batch_size=4,
        grad_batch_size=4,
        explainability_metrics=explainability_metrics,
        eval_batch_size=batch_size,
    )
    model_explainer = ModelExplainer(config=explanation_task_config)
    model_explainer.run(checkpoint_path=checkpoint_path, total_samples=100)


if __name__ == "__main__":
    fire.Fire(main)
