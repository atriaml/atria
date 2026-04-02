from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any, Literal

import torch
from atria_logger import get_logger
from captum._utils.common import _format_additional_forward_args, _format_inputs
from torchxai.data_types._common import TensorOrTupleOfTensorsGeneric

from atria_insights.explainers._attn._target import AttentionTokenTarget
from atria_insights.explainers._attn._utils import compute_flows
from atria_insights.model_pipelines._forward_wrappers._sequence_forward_wrappers import (
    ExplainableSequenceModelForwardWrapper,
)

logger = get_logger(__name__)


class RawAttentionExplainer:
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
    ) -> torch.Tensor:
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

    # def explain(
    #     self,
    #     inputs: TensorOrTupleOfTensorsGeneric,
    #     feature_keys: list[str],
    #     attention_token_target: AttentionTokenTarget,
    #     additional_forward_args: tuple[Any, ...] | None = None,
    # ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
    #     with torch.no_grad():
    #         attns_dict = self._run_forward(
    #             forward_func=self._model,
    #             inputs=inputs,
    #             additional_forward_args=additional_forward_args,
    #         )

    #         # get the attention per key and reduce it key-wise
    #         explanations = ()
    #         for input, key in zip(_format_inputs(inputs), feature_keys, strict=True):
    #             logger.debug(f"Processing attentions for key: {key}")
    #             assert key in attns_dict, (
    #                 f"Key '{key}' not found in model outputs: {list(attns_dict.keys())}"
    #             )
    #             attns = attns_dict[key]  # tuple of (B, H, L, L) per layer

    #             # convert to (B, Layer, H, L, L)
    #             attns = (
    #                 torch.stack([attn.cpu().detach() for attn in attns])
    #                 .permute(1, 0, 2, 3, 4)
    #                 .double()
    #             )
    #             agg = self._aggregate_layers(attns)
    #             agg_for_target = attention_token_target.select(agg)  # (B, T, L_k)
    #             agg_for_target = agg_for_target.squeeze(1)  # (B, L_k)

    #             # we need to remap any attention xplanations to the input feature space, which is what users expect
    #             # for token-level targets, this is a no-op since the target already selects the correct
    #             # but for example if layout_ids are there, we need to spread the token level scores to the layout level (B, S) -> (B, S, 4) for the 4 layout tokens per input token
    #             if key == "layout_ids":
    #                 # we assume that the layout tokens are in the order of [CLS, SEP, PAD, MASK] and that the target is only selecting the CLS token, so we can just repeat the scores 4 times
    #                 bbox_shape = input.shape[-1]
    #                 agg_for_target = (
    #                     agg_for_target.unsqueeze(-1).expand_as(input) / bbox_shape
    #                 )

    #             if key == "image":
    #                 _, c, h, w = input.shape
    #                 num_patches = agg_for_target.shape[-1]
    #                 grid_size = int(math.sqrt(num_patches))
    #                 patch_size = h // grid_size

    #                 agg_for_target = agg_for_target.view(
    #                     agg_for_target.size(0), grid_size, grid_size
    #                 )
    #                 agg_for_target = agg_for_target.repeat_interleave(
    #                     patch_size, dim=1
    #                 ).repeat_interleave(patch_size, dim=2)
    #                 agg_for_target = agg_for_target.unsqueeze(1).expand(-1, c, -1, -1)
    #                 agg_for_target = agg_for_target / (patch_size * patch_size * c)

    #             # we must make sure that the input and explanation feature shapes match
    #             # this makes sure that the attention explainer outputs ultimately look the same as attribution explainers
    #             # this also makes sure we can visualize and validate them in a exact same fashion
    #             assert agg_for_target.shape == input.shape, (
    #                 f"Shape mismatch between input and explanation for key '{key}': "
    #                 f"input shape: {input.shape}, explanation shape: {agg_for_target.shape}"
    #             )

    #             explanations += (agg_for_target,)

    #         return explanations
    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        feature_keys: list[str],
        attention_token_target: AttentionTokenTarget,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        with torch.no_grad():
            # 1. Extract - each model just returns tuple of attention tensors
            attn_tuples = self._run_forward(
                forward_func=self._model,
                inputs=inputs,
                additional_forward_args=additional_forward_args,
            )

            # 2. Aggregate - same for all models
            aggregated = ()
            for attns in attn_tuples:
                attns = (
                    torch.stack([attn.cpu().detach() for attn in attns])
                    .permute(1, 0, 2, 3, 4)
                    .double()
                )
                agg = self._aggregate_layers(attns)
                agg_for_target = attention_token_target.select(agg).squeeze(1)
                aggregated += (agg_for_target,)

            # 3. Map to feature space - model-specific
            explanations = self._model._model.map_attentions_to_feature_space(
                aggregated, _format_inputs(inputs), feature_keys
            )

            return explanations


class AttentionRolloutExplainer(RawAttentionExplainer):
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


class AttentionFlowExplainer(RawAttentionExplainer):
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
