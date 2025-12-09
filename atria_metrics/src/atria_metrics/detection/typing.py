from dataclasses import dataclass

import torch


@dataclass
class GroundTruthInstances:
    bboxes: torch.Tensor
    labels: torch.Tensor

    def detach(self):
        self.bboxes = self.bboxes.detach()
        self.labels = self.labels.detach()


@dataclass
class PredInstances:
    bboxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor

    def detach(self):
        self.bboxes = self.bboxes.detach()
        self.labels = self.labels.detach()
        self.scores = self.scores.detach()
