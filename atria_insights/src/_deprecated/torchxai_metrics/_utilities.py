from __future__ import annotations

from typing import Any

import torch

from atria_insights.data_types import ModelExplainerOutput

# @dataclasses.dataclass(frozen=True)
# class TorchXAIMetricInput:


# def _default_output_transform_list_of_targets(output: ModelExplainerOutput) -> ModelExplainerOutput:
#     pass

# def _default_output_transform_single_target(output: ModelExplainerOutput) -> ModelExplainerOutput:
#     pass

# def _default_output_transform(output: ModelExplainerOutput) -> ModelExplainerOutput:
#     # if target is a list of list we assume it is a multi-target scenario with varying output sizes per sample
#     # in which case we need to iterate over the samples in the batch
#     is_target_list = isinstance(output.target, list)
#     is_target_list_of_lists = isinstance(output.target, list) and isinstance(
#         output.target[0], list
#     )
#     if is_target_list_of_lists:
#         return _default_output_transform_list_of_targets(output)
#     elif is_target_list:
#         return _default_output_transform_single_target(output)

#     metric_kwargs = {
#         "is_multi_target": is_target_list,
#         "explainer": self._explainer,
#         "constant_shifts": (
#             tuple(  # these are only for input invariance
#                 x.detach()
#                 for x in output.explainer_step_inputs.constant_shifts.values()
#             )
#             if output.explainer_step_inputs.constant_shifts is not None
#             else None
#         ),
#         "input_layer_names": (
#             tuple(x for x in output.explainer_step_inputs.input_layer_names.values())
#             if output.explainer_step_inputs.input_layer_names is not None
#             else None
#         ),  # these are only for input invariance
#         "frozen_features": (
#             output.explainer_step_inputs.frozen_features
#             if output.explainer_step_inputs.frozen_features is not None
#             else None
#         ),
#         "train_baselines": tuple(output.explainer_step_inputs.train_baselines.values())
#         if output.explainer_step_inputs.train_baselines is not None
#         else None,
#         "return_intermediate_results": True,
#         "return_dict": True,
#         "show_progress": True,
#     }

#     possible_args = set(inspect.signature(self._metric_func).parameters)
#     if "explainer" in possible_args:
#         explainer_possible_args = set(
#             inspect.signature(self._explainer.explain).parameters
#         )
#         if "baselines" in explainer_possible_args:
#             if "baselines" in possible_args:
#                 possible_args.remove("baselines")
#             possible_args.add("explainer_baselines")
#         possible_args.update(set(inspect.signature(self._explainer.explain).parameters))

#     if is_target_list_of_lists:
#         batch_size = output.explainer_step_inputs.model_inputs.explained_inputs[
#             next(iter(output.explainer_step_inputs.model_inputs.explained_inputs))
#         ].shape[0]

#         metric_kwargs_list = []
#         for batch_idx in range(batch_size):
#             current_metric_kwargs = {}
#             for k, v in metric_kwargs.items():
#                 try:
#                     if v is None:
#                         current_metric_kwargs[k] = None
#                         continue
#                     if k in [
#                         "forward_func",
#                         "is_multi_target",
#                         "explainer",
#                         "constant_shifts",
#                         "input_layer_names",
#                         "return_intermediate_results",
#                         "return_dict",
#                         "show_progress",
#                         "train_baselines",
#                     ]:
#                         current_metric_kwargs[k] = v
#                         continue

#                     if isinstance(v, tuple):
#                         current_metric_kwargs[k] = tuple(
#                             (
#                                 v_i[batch_idx].unsqueeze(0)
#                                 if v_i[batch_idx] is not None
#                                 else v_i[batch_idx]
#                             )
#                             for v_i in v
#                         )
#                     elif isinstance(v, torch.Tensor):
#                         current_metric_kwargs[k] = v[batch_idx].unsqueeze(0)
#                     else:
#                         if (
#                             k == "frozen_features"
#                         ):  # frozen features is a list of tensors
#                             current_metric_kwargs[k] = v[batch_idx].unsqueeze(0)
#                         else:
#                             current_metric_kwargs[k] = v[batch_idx]
#                 except Exception as e:
#                     logger.exception(
#                         f"An error occurred while preparing metric kwargs {k}: {v} for multi-target scenario. Error: {e}"
#                     )
#                     exit(1)

#             total_targets = len(current_metric_kwargs["target"])
#             assert all(
#                 tuple(
#                     x is None or x.shape[1] == len(current_metric_kwargs["target"])
#                     for x in current_metric_kwargs["attributions"]
#                 )
#             ), (
#                 "dim=1 of attributions must have the same size as the total number of targets for each input tuple in multi-target scenario"
#                 f"dim=1 of attributions: {[x.shape[1] for x in current_metric_kwargs['attributions']]}"
#                 f"total number of targets: {total_targets}"
#             )

#             # convert explanations to list of tensors
#             # current_metric_kwargs["attributions"] can be a tuple of None's in case the explanation does not exist
#             # or targets are empty something like (None, None, ...)
#             if current_metric_kwargs["attributions"][0] is not None:
#                 current_metric_kwargs["attributions"] = [
#                     tuple(x[:, t] for x in current_metric_kwargs["attributions"])
#                     for t in range(total_targets)
#                 ]
#             else:
#                 current_metric_kwargs["attributions"] = []
#             assert len(current_metric_kwargs["attributions"]) == total_targets, (
#                 "The number of targets must be equal to the number of attributions as input to the metric function."
#             )
#             metric_kwargs_list += [current_metric_kwargs]
#         metric_kwargs = [
#             {key: value for key, value in metric_kwargs.items() if key in possible_args}
#             for metric_kwargs in metric_kwargs_list
#         ]
#         return metric_kwargs
#     else:
#         metric_kwargs = {
#             key: value for key, value in metric_kwargs.items() if key in possible_args
#         }
#         return metric_kwargs


def _as_detached_tuple(tensor_or_mapping: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(
        x.detach() if isinstance(x, torch.Tensor) else x
        for x in tensor_or_mapping.values()
    )


class ModelExplainerOutputTransform:
    def __init__(self, output: ModelExplainerOutput):
        self.output = output

    @property
    def inputs(self) -> tuple:
        return _as_detached_tuple(
            self.output.explainer_step_inputs.model_inputs.explained_features
        )

    @property
    def additional_forward_args(self) -> tuple | None:
        if (
            self.output.explainer_step_inputs.model_inputs.additional_forward_kwargs
            is None
        ):
            return None
        return _as_detached_tuple(
            self.output.explainer_step_inputs.model_inputs.additional_forward_kwargs
        )

    @property
    def target(self) -> torch.Tensor:
        return self.output.target.detach()

    @property
    def attributions(self) -> tuple[torch.Tensor, ...]:
        return _as_detached_tuple(self.output.explanations)

    @property
    def metric_baselines(self) -> tuple[torch.Tensor, ...] | None:
        if self.output.explainer_step_inputs.metric_baselines is None:
            return None
        return _as_detached_tuple(self.output.explainer_step_inputs.metric_baselines)

    @property
    def explainer_baselines(self) -> tuple[torch.Tensor, ...] | None:
        if self.output.explainer_step_inputs.baselines is None:
            return None
        return _as_detached_tuple(self.output.explainer_step_inputs.baselines)

    @property
    def feature_mask(self) -> tuple[torch.Tensor, ...]:
        if self.output.explainer_step_inputs.feature_masks is None:
            raise ValueError("Feature masks are not provided in the output.")
        return _as_detached_tuple(self.output.explainer_step_inputs.feature_masks)

    @property
    def frozen_features(self) -> list[torch.Tensor] | None:
        return (
            [x.detach() for x in self.output.explainer_step_inputs.frozen_features]
            if self.output.explainer_step_inputs.frozen_features is not None
            else None
        )

    @property
    def constant_shifts(self) -> tuple[torch.Tensor, ...] | None:
        if self.output.explainer_step_inputs.constant_shifts is None:
            return None
        return _as_detached_tuple(self.output.explainer_step_inputs.constant_shifts)

    @property
    def input_layer_names(self) -> tuple[str, ...] | None:
        if self.output.explainer_step_inputs.input_layer_names is None:
            return None
        return tuple(
            x for x in self.output.explainer_step_inputs.input_layer_names.values()
        )
