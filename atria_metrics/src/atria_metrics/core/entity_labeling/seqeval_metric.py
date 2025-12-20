from collections.abc import Mapping

import torch
from atria_models.core.types.model_outputs import TokenClassificationModelOutput
from ignite.engine import Engine
from ignite.metrics.metric import Metric, reinit__is_reduced


class SeqEvalMetric(Metric):
    def __init__(
        self, device: str | torch.device = torch.device("cpu"), scheme: str = "IOB2"
    ) -> None:
        self._check_done = False
        self._scheme = scheme
        super().__init__(device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._total_target_label_names: list[list[str]] = []
        self._total_predicted_label_names: list[list[str]] = []

    @reinit__is_reduced
    def update(self, model_output: TokenClassificationModelOutput) -> None:
        assert model_output.target_label_names is not None, (
            "target_label_names cannot be None. "
            "Ensure the model output includes target label names for evaluation."
        )
        assert model_output.predicted_label_names is not None, (
            "predicted_label_names cannot be None. "
            "Ensure the model output includes predicted label names for evaluation."
        )
        self._total_target_label_names.extend(model_output.target_label_names)
        self._total_predicted_label_names.extend(model_output.predicted_label_names)

    def compute(self) -> float:
        from seqeval.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )

        scores = {
            "accuracy_score": accuracy_score(
                self._total_target_label_names, self._total_predicted_label_names
            ),
            "precision_score": precision_score(
                self._total_target_label_names,
                self._total_predicted_label_names,
                scheme=self._scheme,
            ),
            "recall_score": recall_score(
                self._total_target_label_names,
                self._total_predicted_label_names,
                scheme=self._scheme,
            ),
            "f1_score": f1_score(
                self._total_target_label_names,
                self._total_predicted_label_names,
                scheme=self._scheme,
            ),
            "classification_report": classification_report(
                self._total_target_label_names,
                self._total_predicted_label_names,
                scheme=self._scheme,
                output_dict=True,
            ),
        }
        return scores

    def completed(self, engine: Engine, name: str) -> None:
        result = self.compute()
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(
                    f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}"
                )

            for key, value in result.items():
                engine.state.metrics[name + "/" + key] = value
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[name] = result
