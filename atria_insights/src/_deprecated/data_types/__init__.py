# import dataclasses
# from typing import Any

# import torch


# @dataclasses.dataclass
# class ModelInputs:
#     explained_features: dict[str, torch.Tensor]
#     additional_forward_kwargs: dict[str, Any]


# @dataclasses.dataclass(frozen=True)
# class ExplainerStepInputs:
#     model_inputs: ModelInputs
#     baselines: dict[str, torch.Tensor] | None = None
#     metric_baselines: dict[str, torch.Tensor] | None = None
#     feature_masks: dict[str, torch.Tensor] | None = None
#     constant_shifts: dict[str, torch.Tensor] | None = None
#     input_layer_names: dict[str, str] | None = None
#     train_baselines: dict[str, torch.Tensor] | None = None
#     frozen_features: list[torch.Tensor] | None = None


# @dataclasses.dataclass(frozen=True)
# class ModelExplainerOutput:
#     # sample metadata
#     sample_id: list[str]

#     # sample inputs
#     explainer_step_inputs: ExplainerStepInputs
#     target: torch.Tensor
#     model_outputs: torch.Tensor

#     # explanations
#     explanations: dict[str, torch.Tensor]
#     reduced_explanations: dict[str, torch.Tensor]

#     def __post_init__(self):
#         # validate that explanations and reduced_explanations have the same keys
#         if set(self.explanations.keys()) != set(self.reduced_explanations.keys()):
#             raise ValueError(
#                 "Explanations and reduced_explanations must have the same keys"
#             )

#         if set(
#             self.explainer_step_inputs.model_inputs.explained_features.keys()
#         ) != set(self.explanations.keys()):
#             raise ValueError(
#                 "Explained features and explanations must have the same keys"
#             )

#         assert len(self.sample_id) == self.target.shape[0]
#         assert len(self.sample_id) == self.model_outputs.shape[0]
#         if self.explainer_step_inputs.frozen_features is not None:
#             assert len(self.sample_id) == len(
#                 self.explainer_step_inputs.frozen_features
#             )


# @dataclasses.dataclass(frozen=True)
# class MultiTargetModelExplainerOutput:
#     target: list[torch.Tensor]
#     explanations: list[dict[str, torch.Tensor]]
#     reduced_explanations: list[dict[str, torch.Tensor]]


# @dataclasses.dataclass(frozen=True)
# class ImageClassificationExplainerStepOutput(ModelExplainerOutput):
#     # additional outputs
#     prediction_probs: torch.Tensor | None = None
#     gt_label_value: torch.Tensor | None = None
#     gt_label_name: list[str] | None = None
#     predicted_label_value: torch.Tensor | None = None
#     predicted_label_name: list[str] | None = None
