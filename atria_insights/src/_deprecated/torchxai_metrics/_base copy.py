"""TorchXAI metric implementation for model explanation evaluation."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from atria_logger import get_logger
from ignite.handlers import ProgressBar
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.utils import apply_to_tensor

from atria_insights.utilities.containers import ExplainerStepOutput

if TYPE_CHECKING:
    from torchxai.explainers.explainer import Explainer

logger = get_logger(__name__)


def default_output_transform(output: ExplainerStepOutput) -> ExplainerStepOutput:
    """Validate and return the model output.

    Args:
        output: The model output to validate.

    Returns:
        The validated output.

    Raises:
        AssertionError: If output is not an ExplainerStepOutput instance.
    """
    if not isinstance(output, ExplainerStepOutput):
        msg = (
            f"The output of the model must be an instance of "
            f"ExplainerStepOutput, got: {type(output)}"
        )
        raise TypeError(msg)
    return output


def detach(x: Any) -> Any:
    """Recursively detach tensors from computation graph.

    Args:
        x: Input to detach (tensor, list, or other type).

    Returns:
        Detached version of input.
    """
    if isinstance(x, list):
        return [detach(item) if item is not None else None for item in x]
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


class TorchXAIMetricBase(Metric):
    def __init__(
        self,
        metric_func: partial[Callable],
        forward_func: torch.nn.Module,
        explainer: Explainer,
        output_transform: Callable = default_output_transform,
        device: str = "cpu",
        progress_bar: ProgressBar | None = None,
        attached_name: str | None = None,
        with_amp: bool = False,
    ) -> None:
        """Initialize the TorchXAI metric.

        Args:
            metric_func: Partial function wrapping the metric computation.
            forward_func: Model forward function.
            explainer: TorchXAI explainer instance.
            output_transform: Function to transform engine output.
            device: Device to use for computation.
            progress_bar: Optional progress bar for tracking.
            attached_name: Optional name for the attached metric.
        """
        self._attached_name = attached_name
        self._metric_func = metric_func
        self._metric_name = self._metric_func.func.__name__
        self._forward_func = forward_func
        self._explainer = explainer
        self._metric_outputs: list[dict[str, Any]] = []
        self._progress_bar = progress_bar
        self._with_amp = with_amp
        self._num_examples = 0
        self._result: float | None = None
        super().__init__(output_transform=output_transform, device=device)

    @property
    def metric_name(self) -> str:
        """Get the metric function name."""
        return self._metric_name

    @reinit__is_reduced
    def reset(self) -> None:
        """Reset metric state for new epoch."""
        self._metric_outputs = []
        self._num_examples = 0
        self._result = None
        super().reset()

    def _prepare_metric_kwargs(
        self, output: ExplainerStepOutput
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Prepare keyword arguments for metric computation.

        Args:
            output: Explainer step output containing all necessary data.

        Returns:
            Dictionary or list of dictionaries with metric kwargs.
        """
        is_target_list = isinstance(output.target, list)
        is_target_list_of_lists = is_target_list and isinstance(output.target[0], list)

        # Build base metric kwargs
        metric_kwargs = self._build_base_metric_kwargs(output, is_target_list)

        # Get valid parameter names
        possible_args = self._get_possible_args()

        # Handle multi-target list-of-lists case
        if is_target_list_of_lists:
            return self._prepare_multi_target_kwargs(
                output, metric_kwargs, possible_args
            )

        # Filter to only valid arguments
        return {k: v for k, v in metric_kwargs.items() if k in possible_args}

    def _build_base_metric_kwargs(
        self, output: ExplainerStepOutput, is_target_list: bool
    ) -> dict[str, Any]:
        """Build base metric keyword arguments."""
        model_inputs = output.explainer_step_inputs.model_inputs

        return {
            "forward_func": self._forward_func,
            "inputs": tuple(x.detach() for x in model_inputs.explained_inputs.values()),
            "additional_forward_args": (
                tuple(
                    x.detach() for x in model_inputs.additional_forward_kwargs.values()
                )
                if model_inputs.additional_forward_kwargs
                else None
            ),
            "target": (
                output.target.detach()
                if isinstance(output.target, torch.Tensor)
                else output.target
            ),
            "attributions": tuple(detach(x) for x in output.explanations.values()),
            "baselines": self._get_optional_tuple(
                output.explainer_step_inputs.metric_baselines
            ),
            "explainer_baselines": self._get_optional_tuple(
                output.explainer_step_inputs.baselines
            ),
            "feature_mask": tuple(
                x.detach() for x in output.explainer_step_inputs.feature_masks.values()
            ),
            "is_multi_target": is_target_list,
            "explainer": self._explainer,
            "constant_shifts": self._get_optional_tuple(
                output.explainer_step_inputs.constant_shifts
            ),
            "input_layer_names": (
                tuple(output.explainer_step_inputs.input_layer_names.values())
                if output.explainer_step_inputs.input_layer_names is not None
                else None
            ),
            "frozen_features": output.explainer_step_inputs.frozen_features,
            "train_baselines": (
                tuple(output.explainer_step_inputs.train_baselines.values())
                if output.explainer_step_inputs.train_baselines is not None
                else None
            ),
            "return_intermediate_results": True,
            "return_dict": True,
            "show_progress": True,
        }

    @staticmethod
    def _get_optional_tuple(attr_dict: dict | None) -> tuple | None:
        """Get tuple from dict if all values are not None."""
        if attr_dict is None:
            return None
        if all(x is not None for x in attr_dict.values()):
            return tuple(x.detach() for x in attr_dict.values())
        return None

    def _get_possible_args(self) -> set[str]:
        """Get set of possible argument names for the metric function."""
        possible_args = set(inspect.signature(self._metric_func).parameters)

        if "explainer" in possible_args:
            explainer_params = set(
                inspect.signature(self._explainer.explain).parameters
            )
            if "baselines" in explainer_params and "baselines" in possible_args:
                possible_args.remove("baselines")
                possible_args.add("explainer_baselines")
            possible_args.update(explainer_params)

        return possible_args

    def _prepare_multi_target_kwargs(
        self,
        output: ExplainerStepOutput,
        metric_kwargs: dict[str, Any],
        possible_args: set[str],
    ) -> list[dict[str, Any]]:
        """Prepare kwargs for multi-target scenario."""
        batch_size = output.explainer_step_inputs.model_inputs.explained_inputs[
            next(iter(output.explainer_step_inputs.model_inputs.explained_inputs))
        ].shape[0]

        static_keys = {
            "forward_func",
            "is_multi_target",
            "explainer",
            "constant_shifts",
            "input_layer_names",
            "return_intermediate_results",
            "return_dict",
            "show_progress",
            "train_baselines",
        }

        metric_kwargs_list = []
        for batch_idx in range(batch_size):
            current_kwargs = self._extract_batch_kwargs(
                metric_kwargs, batch_idx, static_keys
            )
            self._validate_and_split_attributions(current_kwargs)
            metric_kwargs_list.append(current_kwargs)

        return [
            {k: v for k, v in kwargs.items() if k in possible_args}
            for kwargs in metric_kwargs_list
        ]

    def _extract_batch_kwargs(
        self, metric_kwargs: dict[str, Any], batch_idx: int, static_keys: set[str]
    ) -> dict[str, Any]:
        """Extract kwargs for a single batch item."""
        current_kwargs = {}

        for key, value in metric_kwargs.items():
            if value is None or key in static_keys:
                current_kwargs[key] = value
                continue

            try:
                if isinstance(value, tuple):
                    current_kwargs[key] = tuple(
                        v_i[batch_idx].unsqueeze(0)
                        if v_i[batch_idx] is not None
                        else v_i[batch_idx]
                        for v_i in value
                    )
                elif isinstance(value, torch.Tensor):
                    current_kwargs[key] = value[batch_idx].unsqueeze(0)
                elif key == "frozen_features":
                    current_kwargs[key] = value[batch_idx].unsqueeze(0)
                else:
                    current_kwargs[key] = value[batch_idx]
            except Exception:
                logger.exception(
                    f"Error preparing metric kwargs {key}: {value} for multi-target"
                )
                raise

        return current_kwargs

    def _validate_and_split_attributions(self, metric_kwargs: dict[str, Any]) -> None:
        """Validate and split attributions for multi-target case."""
        total_targets = len(metric_kwargs["target"])
        attributions = metric_kwargs["attributions"]

        # Validate attribution dimensions
        if attributions[0] is not None:
            for attr in attributions:
                if attr is not None and attr.shape[1] != total_targets:
                    msg = (
                        f"dim=1 of attributions must match total targets. "
                        f"Got {attr.shape[1]}, expected {total_targets}"
                    )
                    raise ValueError(msg)

            # Split attributions by target
            metric_kwargs["attributions"] = [
                tuple(x[:, t] for x in attributions) for t in range(total_targets)
            ]
        else:
            metric_kwargs["attributions"] = []

        if len(metric_kwargs["attributions"]) != total_targets:
            msg = (
                f"Number of attributions ({len(metric_kwargs['attributions'])}) "
                f"must equal number of targets ({total_targets})"
            )
            raise ValueError(msg)

    @reinit__is_reduced
    def update(self, output: ExplainerStepOutput) -> None:
        """Update metric with new batch output.

        Args:
            output: Explainer step output for the current batch.
        """
        if output.explanations is None:
            return

        self._log_metric_computation()

        start_time = time.time()
        metric_kwargs = self._prepare_metric_kwargs(output)
        metric_output = self._compute_metric(metric_kwargs)
        end_time = time.time()

        # Add timing information
        time_taken = torch.tensor(end_time - start_time, requires_grad=False)
        metric_output["time_taken"] = torch.stack(
            [time_taken / len(output.index) for _ in range(len(output.index))]
        )

        self._num_examples += len(metric_output)
        self._metric_outputs.append(metric_output)

    def _log_metric_computation(self) -> None:
        """Log metric computation start."""
        if self._attached_name is not None:
            logger.info(f"Computing metric=[{self._attached_name}.{self._metric_name}]")
        else:
            logger.info(f"Computing metric=[{self._metric_name}]")

    def _compute_metric(
        self, metric_kwargs: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute metric from kwargs."""
        from torch.cuda.amp.autocast_mode import autocast

        if isinstance(metric_kwargs, list):
            return self._compute_multi_sample_metric(metric_kwargs)

        with autocast(enabled=self._with_amp):
            logger.info(f"Computing metric: {self._metric_func}")
            metric_output = self._metric_func(**metric_kwargs)

        return apply_to_tensor(metric_output, lambda tensor: tensor.detach().cpu())

    def _compute_multi_sample_metric(
        self, metric_kwargs_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute metric for multiple samples."""
        metric_outputs = []

        for kwargs in metric_kwargs_list:
            if "target" in kwargs and len(kwargs["target"]) == 0:
                metric_outputs.append({"failure": True})
                continue

            output = self._metric_func(**kwargs)
            output = apply_to_tensor(output, lambda tensor: tensor.detach().cpu())
            metric_outputs.append(output)

        # Aggregate outputs
        return {
            key: [d.get(key, -1000) for d in metric_outputs]
            for key in metric_outputs[0].keys()
        }

    @sync_all_reduce("_num_examples")
    def compute(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Compute final metric value.

        Returns:
            Dictionary or list of dictionaries containing metric results.

        Raises:
            RuntimeError: If metric computation fails.
        """
        if self._num_examples == 0:
            return {}

        try:
            return self._metric_outputs
        except Exception as exc:
            logger.exception(f"Error computing metric {self._metric_name}: {exc}")
            raise RuntimeError(f"Failed to compute metric {self._metric_name}") from exc
