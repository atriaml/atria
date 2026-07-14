# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Membership inference attack over precomputed feature vectors.

``MembershipInferenceBlackBox`` below is ART's own class of the same name
(art.attacks.inference.membership_inference.black_box), copied and stripped to only the
path we actually use: prediction-based input, no labels, no target-model estimator.
ART's original requires a real target-model estimator purely to read
``nb_classes``/``input_shape`` off of, even though it's never called (we always supply
``members_x=``/``non_members_x=`` directly) -- and for ``attack_model_type="nn"`` with no labels,
it goes further and reuses ``estimator.nb_classes`` as the width of its attack network's
input layer, which only works if that happens to equal the feature width. We don't have
an estimator at all: the "predictions" are ``TokenSignalExtractor``'s engineered feature
vectors, and we only ever have one input type (features), so the estimator/nb_classes/
input_shape machinery, the raw x/y record mode, the loss-based input_type, the regressor
mode, and the with-label mode are all removed. Everything else (scaling,
per-attack_model_type training including the real PyTorch network for "nn") is
unmodified from ART.

``MembershipInferenceAttack`` is our own wrapper around it -- not part of ART -- that
``ModelAttacker`` calls directly: it builds a ``MembershipInferenceBlackBox`` from
``AttackConfig``, runs it end-to-end over the four feature-DataFrame splits, and reports
accuracy/AUC/ROC/worst-case-TPR@FPR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from atria_logger import get_logger

from atria_prv.att._attacks._evaluation import _accuracy_report, _tpr_at_targeted_fpr

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.base import ClassifierMixin

    from atria_prv.att.configs import AttackConfig

logger = get_logger(__name__)


class MembershipInferenceBlackBox:
    """Learned black-box membership inference attack over feature vectors.

    The attack model (rf / gb / lr / dt / knn / svm / nn) is fit on member vs
    non-member feature vectors and can then infer membership (or probabilities) for
    held-out feature vectors.
    """

    def __init__(
        self,
        attack_model_type: str = "nn",
        attack_model: ClassifierMixin | None = None,
        scaler_type: str | None = "standard",
        nn_model_epochs: int = 500,
        nn_model_batch_size: int = 16,
        nn_model_learning_rate: float = 0.0001,
    ) -> None:
        self.attack_model_type = attack_model_type
        self.attack_model = attack_model
        self.scaler_type = scaler_type
        self.scaler: Any | None = None
        self.epochs = nn_model_epochs
        self.batch_size = nn_model_batch_size
        self.learning_rate = nn_model_learning_rate

        self._check_params()

        if self.attack_model:
            self.default_model = False
        else:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                RandomForestClassifier,
            )
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeClassifier

            self.default_model = True
            if self.attack_model_type == "rf":
                self.attack_model = RandomForestClassifier()
            elif self.attack_model_type == "gb":
                self.attack_model = GradientBoostingClassifier()
            elif self.attack_model_type == "lr":
                self.attack_model = LogisticRegression()
            elif self.attack_model_type == "dt":
                self.attack_model = DecisionTreeClassifier()
            elif self.attack_model_type == "knn":
                self.attack_model = KNeighborsClassifier()
            elif self.attack_model_type == "svm":
                self.attack_model = SVC(probability=True)
            elif self.attack_model_type != "nn":
                raise ValueError("Illegal value for parameter `attack_model_type`.")

    def _check_params(self) -> None:
        if self.attack_model_type not in ["nn", "rf", "gb", "lr", "dt", "knn", "svm"]:
            raise ValueError("Illegal value for parameter `attack_model_type`.")
        if self.attack_model:
            from sklearn.base import ClassifierMixin

            if ClassifierMixin not in type(self.attack_model).__mro__:
                raise TypeError("Attack model must be of type Classifier.")

    def fit(self, *, members_x: np.ndarray, non_members_x: np.ndarray) -> None:
        """Train the attack model on member (``members_x``) vs non-member (``non_members_x``) features."""
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        features = members_x.astype(np.float32)
        non_member_features = non_members_x.astype(np.float32)

        labels = np.ones(len(members_x))
        non_member_labels = np.zeros(len(non_members_x))

        x_1 = np.concatenate((features, non_member_features))
        y_new = np.concatenate((labels, non_member_labels))

        if self.scaler_type:
            if self.scaler_type == "standard":
                self.scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaler_type == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError("Illegal scaler_type: ", self.scaler_type)

        if self.default_model and self.attack_model_type == "nn":
            if self.scaler:
                self.scaler.fit(x_1)
                x_1 = self.scaler.transform(x_1)

            num_features = x_1.shape[1]

            # class MembershipInferenceAttackModelNoLabel(nn.Module):
            #     """PyTorch model for learning a membership inference attack from features alone."""

            #     def __init__(self, num_features):
            #         self.num_features = num_features
            #         super().__init__()
            #         self.features = nn.Sequential(
            #             nn.Linear(self.num_features, 512),
            #             nn.ReLU(),
            #             nn.Linear(512, 100),
            #             nn.ReLU(),
            #             nn.Linear(100, 64),
            #             nn.ReLU(),
            #             nn.Linear(64, 1),
            #         )
            #         self.output = nn.Sigmoid()

            #     def forward(self, x_1):
            #         out_x1 = self.features(x_1)
            #         return self.output(out_x1)
            from sklearn.neural_network import MLPClassifier

            self.attack_model = MLPClassifier(
                hidden_layer_sizes=(32, 16), max_iter=500, random_state=0
            )
            self.attack_model.fit(x_1, y_new)
            # print("self.attack_model", self.attack_model)

            # loss_fn = nn.BCELoss()
            # optimizer = optim.SGD(self.attack_model.parameters(), lr=0.01)

            # attack_train_set = self._get_attack_dataset(f_1=x_1, label=y_new)
            # train_loader = DataLoader(
            #     attack_train_set,
            #     batch_size=self.batch_size,
            #     shuffle=True,
            #     num_workers=0,
            # )

            # self.attack_model = to_cuda(self.attack_model)
            # self.attack_model.train()

            # for epoch in range(self.epochs):
            #     avg_loss = 0.0
            #     n_steps = 0
            #     for input1, targets in train_loader:
            #         input1, targets = to_cuda(input1), to_cuda(targets)
            #         input1 = torch.autograd.Variable(input1)
            #         targets = torch.autograd.Variable(targets)

            #         optimizer.zero_grad()
            #         outputs = self.attack_model(input1)
            #         loss = loss_fn(outputs, targets.unsqueeze(1))
            #         loss.backward()
            #         optimizer.step()

            #         avg_loss += loss.item()
            #         n_steps += 1
            #     avg_loss = avg_loss / n_steps
            #     if epoch % 10 == 0:
            #         print(f"Loss = {avg_loss}")
            # exit()
        else:  # not nn
            y_ready = y_new.astype(np.int64)
            if self.scaler:
                self.scaler.fit(x_1)
                x_1 = self.scaler.transform(x_1)
            self.attack_model.fit(x_1, y_ready.ravel())

    def infer(self, *, x: np.ndarray, probabilities: bool = False) -> np.ndarray:
        """Infer membership status (or probabilities) for feature vectors ``x``."""
        x = x.astype(np.float32)

        # if self.default_model and self.attack_model_type == "nn":
        #     import torch
        #     from art.utils import from_cuda, to_cuda
        #     from torch.utils.data import DataLoader

        #     if self.scaler:
        #         x = self.scaler.transform(x)

        #     # self.attack_model.eval()
        #     predictions: np.ndarray | None = None

        #     test_set = self._get_attack_dataset(f_1=x)
        #     test_loader = DataLoader(
        #         test_set, batch_size=self.batch_size, shuffle=False, num_workers=0
        #     )
        #     for input1, _ in test_loader:
        #         input1 = to_cuda(input1)
        #         outputs = self.attack_model(input1)
        #         predicted = outputs if probabilities else torch.round(outputs)
        #         predicted = from_cuda(predicted)

        #         if predictions is None:
        #             predictions = predicted.detach().numpy()
        #         else:
        #             predictions = np.vstack((predictions, predicted.detach().numpy()))

        #     if predictions is None:  # pragma: no cover
        #         raise ValueError("No data available.")
        #     inferred_return = predictions if probabilities else np.round(predictions)
        # elif not self.default_model:
        # assumes the supplied model's predict() returns probabilities
        if self.scaler:
            x = self.scaler.transform(x)
        inferred = self.attack_model.predict_proba(x)
        inferred_return = (
            inferred[:, [1]] if probabilities else np.round(inferred[:, [1]])
        )

        return inferred_return

    def _get_attack_dataset(self, f_1, label=None):
        # used by the "nn" attack_model_type: wraps the feature matrix into a PyTorch
        # Dataset for DataLoader batching (both training in fit() and scoring in infer())
        from torch.utils.data.dataset import Dataset

        class AttackDataset(Dataset):
            """PyTorch dataset wrapping feature vectors (and optional membership labels)."""

            def __init__(self, x_1, y=None):
                import torch

                self.x_1 = torch.from_numpy(x_1.astype(np.float64)).type(
                    torch.FloatTensor
                )
                if y is not None:
                    self.y = torch.from_numpy(y.astype(np.int8)).type(torch.FloatTensor)
                else:
                    self.y = torch.zeros(x_1.shape[0])

            def __len__(self):
                return len(self.x_1)

            def __getitem__(self, idx):
                if idx >= len(self.x_1):  # pragma: no cover
                    raise IndexError("Invalid Index")
                return self.x_1[idx], self.y[idx]

        return AttackDataset(x_1=f_1, y=label)


class MembershipInferenceAttack:
    """Runs a ``MembershipInferenceBlackBox`` end-to-end and reports results.

    Not part of ART -- this is the class ``ModelAttacker`` calls directly.
    """

    def __init__(self, cfg: AttackConfig) -> None:
        self._cfg = cfg
        self._attack = MembershipInferenceBlackBox(
            attack_model_type=cfg.attack_model_type
        )

    def run(
        self,
        *,
        features_members_train: pd.DataFrame,
        features_nonmembers_train: pd.DataFrame,
        features_members_test: pd.DataFrame,
        features_nonmembers_test: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        from sklearn.metrics import roc_auc_score, roc_curve

        def to_matrix(df: pd.DataFrame) -> np.ndarray:
            columns = feature_columns or list(df.columns)
            return df[columns].to_numpy(dtype=np.float32)

        # fit on the attack-train halves; features are the only input, no labels
        print(
            "to_matrix(features_members_train)",
            to_matrix(features_members_train).tolist(),
        )
        print(
            "to_matrix(features_nonmembers_train)",
            to_matrix(features_nonmembers_train).tolist(),
        )
        self._attack.fit(
            members_x=to_matrix(features_members_train),
            non_members_x=to_matrix(features_nonmembers_train),
        )

        # score the held-out attack-test halves
        inferred_members = self._attack.infer(
            x=to_matrix(features_members_test)
        )  # want 1s
        inferred_nonmembers = self._attack.infer(
            x=to_matrix(features_nonmembers_test)
        )  # want 0s
        report = _accuracy_report(inferred_members, inferred_nonmembers)

        # AUC + worst-case TPR@FPR on the attack-test halves
        prob_members = np.squeeze(
            self._attack.infer(x=to_matrix(features_members_test), probabilities=True),
            axis=-1,
        )
        prob_nonmembers = np.squeeze(
            self._attack.infer(
                x=to_matrix(features_nonmembers_test), probabilities=True
            ),
            axis=-1,
        )
        attack_proba = np.concatenate([prob_members, prob_nonmembers])
        attack_true = np.concatenate(
            [np.ones(len(prob_members)), np.zeros(len(prob_nonmembers))]
        )
        report["auc"] = float(roc_auc_score(attack_true, attack_proba))

        roc_fpr, roc_tpr, roc_thresholds = roc_curve(attack_true, attack_proba)
        report["roc_curve"] = {"fpr": roc_fpr.tolist(), "tpr": roc_tpr.tolist()}

        achieved_fpr, achieved_tpr, threshold = _tpr_at_targeted_fpr(
            roc_fpr, roc_tpr, roc_thresholds, self._cfg.targeted_fpr
        )
        report["worst_case"] = {
            "targeted_fpr": self._cfg.targeted_fpr,
            "tpr": achieved_tpr,
            "fpr": achieved_fpr,
            "threshold": threshold,
        }

        logger.info(
            f"Membership inference attack ({self._attack.attack_model_type}): {report}"
        )
        return report
