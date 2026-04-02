from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any, Literal

import torch
from atria_logger import get_logger
from captum._utils.common import _format_additional_forward_args, _format_inputs
from torchxai.data_types._common import TensorOrTupleOfTensorsGeneric
from torchxai.explainers._explainer import Explainer

from atria_insights.explainers._attn._target import BatchAttentionTokenTarget
from atria_insights.explainers._attn._utils import compute_flows
from atria_insights.model_pipelines._forward_wrappers._sequence_forward_wrappers import (
    ExplainableSequenceModelForwardWrapper,
)

logger = get_logger(__name__)


class AttentionExplainer(Explainer):
    """
    Generic attention explainer for TransformersEncoderModel-based models.
    """

    def __init__(
        self,
        model: ExplainableSequenceModelForwardWrapper,
        head_reduction: Literal["mean", "max", "min", "sum"] = "mean",
    ) -> None:
        self._model = model
        self._head_reduction = head_reduction

    def _reduce_heads(self, attn: torch.Tensor) -> torch.Tensor:
        """Reduce attention heads: (B, Layer, H, L, L) -> (B, Layer, L, L)."""
        match self._head_reduction:
            case "mean":
                return attn.mean(dim=2)
            case "max":
                return attn.max(dim=2).values
            case "min":
                return attn.min(dim=2).values
            case "sum":
                return attn.sum(dim=2)
            case _:
                raise ValueError(
                    f"Unsupported head reduction strategy: {self._head_reduction}"
                )

    def _aggregate_layers(self, attentions: torch.Tensor) -> torch.Tensor:
        """Aggregate per-layer attentions into a single (B, L, L) tensor."""
        # get the attentions of the last layer as default aggregation
        return self._reduce_heads(attentions)[:, -1]

    def _run_forward(
        self, forward_func: Callable, inputs: Any, additional_forward_args: Any
    ) -> tuple[torch.Tensor, ...]:
        forward_func_args = signature(self._model).parameters
        if len(forward_func_args) == 0:
            return self._model()

        inputs = _format_inputs(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        output = forward_func(
            *(*inputs, *additional_forward_args)
            if additional_forward_args is not None
            else inputs
        )
        return output

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        feature_keys: list[str],
        attention_token_target: BatchAttentionTokenTarget,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        with torch.no_grad():
            self._model.return_attns = True
            # 1. Extract - each model just returns tuple of attention tensors
            attn_tuples = self._run_forward(
                forward_func=self._model,
                inputs=inputs,
                additional_forward_args=additional_forward_args,
            )
            self._model.return_attns = False

            # 2. Aggregate - same for all models
            selected_attns_tuples = ()
            for attns in attn_tuples:
                attns = (
                    torch.stack([attn.cpu().detach() for attn in attns])
                    .permute(1, 0, 2, 3, 4)
                    .double()
                )
                processed_attn = self._aggregate_layers(attns)

                # this returns a list of attns over each target
                selected_attns = attention_token_target.select(processed_attn)

                # we create a tuple of selected attns for each target, which we will then map to feature space in the next step
                selected_attns_tuples += (selected_attns,)

            # remap tuple of list of targets to list of tuple of targets, which is the format expected by the mapping function
            selected_attns_tuples = list(zip(*selected_attns_tuples, strict=True))

            # 3. Map to feature space - model-specific for each target
            explanations_per_target = []
            for selected_attns_tuple in selected_attns_tuples:
                explanations = self._model._model.map_attentions_to_feature_space(
                    selected_attns_tuple, _format_inputs(inputs), feature_keys
                )
                explanations_per_target.append(explanations)

            if len(explanations_per_target) == 1:
                return explanations_per_target[0]
            return explanations_per_target

    def __repr__(self) -> str:
        attr_str = ", ".join(
            f"{attr.lstrip('_')}={getattr(self, attr)}" for attr in self.__repr_attrs__
        )
        return f"{self.__class__.__name__}({attr_str})"


class AttentionRolloutExplainer(AttentionExplainer):
    def _aggregate_layers(self, attentions: torch.Tensor) -> torch.Tensor:
        """Aggregate per-layer attentions using attention rollout."""
        # first we reduce over heads
        reduced_attentions = self._reduce_heads(attentions)  # (B, Layer, L, L)

        # add identity to account for residual connections and normalize
        reduced_attentions = (
            reduced_attentions + torch.eye(reduced_attentions.shape[2])[None, None, ...]
        )
        reduced_attentions = (
            reduced_attentions / reduced_attentions.sum(dim=-1)[..., None]
        )

        # compute rollout as done in compute_joint_attention https://github.com/samiraabnar/attention_flow/blob/master/attention_graph_util.py
        rollout = torch.zeros(reduced_attentions.shape, dtype=reduced_attentions.dtype)
        layers = rollout.shape[1]
        rollout[:, 0] = reduced_attentions[:, 0]
        for idx in range(1, layers):
            rollout[:, idx] = reduced_attentions[:, idx].bmm(rollout[:, idx - 1])
        return rollout[:, -1]


class AttentionFlowExplainer(AttentionExplainer):
    """This implementation is correct but extremely expensive as the number of tokens grow, since
    we need to compute max-flow from each node in the attention graph to the input tokens, which is O(N^2) in the number of tokens,
    and we need to do this for each layer and each example in the batch."""

    def _aggregate_layers(self, attentions: torch.Tensor) -> torch.Tensor:
        """Aggregate per-layer attentions using attention rollout."""
        # first we reduce over heads
        reduced_attentions = self._reduce_heads(attentions)  # (B, Layer, L, L)

        # add identity to account for residual connections and normalize
        reduced_attentions = (
            reduced_attentions + torch.eye(reduced_attentions.shape[2])[None, None, ...]
        )
        reduced_attentions = (
            reduced_attentions / reduced_attentions.sum(dim=-1)[..., None]
        )

        batch_attn_flow = []
        for res_att_mat in reduced_attentions:
            example_attn_flow = compute_flows(res_att_mat)
            batch_attn_flow.append(example_attn_flow)
        batch_attn_flow = torch.tensor(batch_attn_flow)
        return batch_attn_flow
