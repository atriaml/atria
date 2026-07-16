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

``MembershipInferenceBlackBox`` below started as ART's class of the same name
(art.attacks.inference.membership_inference.black_box), stripped to the
features-only / no-estimator path we use (see original module docstring for
that history). This version additionally:

- Wraps the (scaler, base classifier) as a single sklearn ``Pipeline`` so
  scaling and model fitting travel together as one estimator.
- Optionally ensembles that pipeline via ``sklearn.ensemble.BaggingClassifier``
  (bootstrap-resampled copies, soft-voted / probability-averaged
  automatically) to reduce variance from any single attack model fit --
  particularly relevant for high-variance base models like the "nn" MLP.
- Optionally runs ``RandomizedSearchCV`` over the base pipeline *before*
  ensembling, to tune the base classifier's hyperparameters on the combined
  member/non-member feature set. The *tuned* pipeline is what then gets
  bagged for the final model.

Backward compatible: with ``n_estimators=1``, ``bootstrap=False``, and
``tune_hyperparameters=False`` (all defaults), this fits a single untouched
copy of the base pipeline -- identical to the original single-model
behavior, just going through the Pipeline/Bagging machinery uniformly.

``MembershipInferenceAttack`` is our own wrapper (not part of ART) that
``ModelAttacker`` calls directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from atria_logger import get_logger

from atria_prv.att._attacks._evaluation import _accuracy_report

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.base import ClassifierMixin
    from sklearn.pipeline import Pipeline

    from atria_prv.att.configs import AttackConfig

logger = get_logger(__name__)

VALID_MODEL_TYPES = ["nn", "rf", "gb", "lr", "dt", "knn", "svm"]


def _default_param_distributions(attack_model_type: str) -> dict[str, Any]:
    """Reasonable default RandomizedSearchCV distributions per attack_model_type.

    Deferred import of scipy.stats so it's only pulled in when tuning is
    actually requested. Override by passing `param_distributions` explicitly
    to `MembershipInferenceBlackBox` if these defaults don't fit your data.
    """
    from scipy.stats import loguniform, randint, uniform

    return {
        "gb": {
            "clf__n_estimators": randint(50, 400),
            "clf__learning_rate": uniform(0.01, 0.29),  # ~0.01-0.30
            "clf__max_depth": randint(2, 6),
            "clf__min_samples_leaf": randint(1, 20),
            "clf__subsample": uniform(0.6, 0.4),  # ~0.6-1.0
        },
        "rf": {
            "clf__n_estimators": randint(50, 400),
            "clf__max_depth": randint(2, 20),
            "clf__min_samples_leaf": randint(1, 20),
            "clf__max_features": uniform(0.3, 0.7),
        },
        "lr": {"clf__C": loguniform(1e-3, 1e2)},
        "dt": {
            "clf__max_depth": randint(2, 20),
            "clf__min_samples_leaf": randint(1, 20),
        },
        "knn": {
            "clf__n_neighbors": randint(3, 30),
            "clf__weights": ["uniform", "distance"],
        },
        "svm": {"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e0)},
        "nn": {
            "clf__hidden_layer_sizes": [
                (32,),
                (64,),
                (32, 16),
                (64, 32),
                (128, 64, 64),
            ],
            "clf__alpha": loguniform(1e-5, 1e-1),
            "clf__learning_rate_init": loguniform(1e-4, 1e-1),
        },
    }[attack_model_type]


class MembershipInferenceBlackBox:
    """Learned black-box membership inference attack over feature vectors.

    The attack model (rf / gb / lr / dt / knn / svm / nn) is fit on member vs
    non-member feature vectors and can then infer membership (or
    probabilities) for held-out feature vectors. Optionally ensembled via
    bagging and/or hyperparameter-tuned via randomized search before the
    final fit.
    """

    def __init__(
        self,
        attack_model_type: str = "nn",
        attack_model: ClassifierMixin | None = None,
        scaler_type: str | None = "robust",
        nn_model_epochs: int = 1000,
        nn_model_batch_size: int = 16,
        nn_model_learning_rate: float = 0.0001,
        # -- ensembling (BaggingClassifier) --
        n_estimators: int = 10,
        bootstrap: bool = True,
        max_samples: float | int = 1.0,
        max_features: float | int = 1.0,
        random_state: int | None = None,
        n_jobs: int | None = -1,
        # -- hyperparameter tuning (RandomizedSearchCV), runs before bagging --
        tune_hyperparameters: bool = False,
        param_distributions: dict[str, Any] | None = None,
        search_n_iter: int = 250,
        search_cv: int = 5,
        search_scoring: str = "roc_auc",
    ) -> None:
        self.attack_model_type = attack_model_type
        self.attack_model = attack_model
        self.scaler_type = scaler_type
        self.epochs = nn_model_epochs
        self.batch_size = nn_model_batch_size
        self.learning_rate = nn_model_learning_rate

        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.tune_hyperparameters = tune_hyperparameters
        self.param_distributions = param_distributions
        self.search_n_iter = search_n_iter
        self.search_cv = search_cv
        self.search_scoring = search_scoring

        self._check_params()
        self.default_model = attack_model is None

        self.model: Any | None = None  # the fitted Pipeline or BaggingClassifier
        self.search_results_: Any | None = None  # RandomizedSearchCV, if tuning ran

    def _check_params(self) -> None:
        if self.attack_model_type not in VALID_MODEL_TYPES:
            raise ValueError("Illegal value for parameter `attack_model_type`.")
        if self.attack_model:
            from sklearn.base import ClassifierMixin

            if ClassifierMixin not in type(self.attack_model).__mro__:
                raise TypeError("Attack model must be of type Classifier.")
        if self.n_estimators < 1:
            raise ValueError("`n_estimators` must be >= 1.")

    def _build_base_model(self) -> Any:
        """Instantiate the base classifier (or return the user-supplied one)."""
        if self.attack_model is not None:
            return self.attack_model

        if self.attack_model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier()
        elif self.attack_model_type == "gb":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier()
        elif self.attack_model_type == "lr":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression()
        elif self.attack_model_type == "dt":
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier()
        elif self.attack_model_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier

            return KNeighborsClassifier()
        elif self.attack_model_type == "svm":
            from sklearn.svm import SVC

            return SVC(probability=True)
        elif self.attack_model_type == "nn":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=self.epochs,
                random_state=None,  # left unseeded on purpose -- see note in fit()
                verbose=False,
            )
        else:  # pragma: no cover - guarded by _check_params
            raise ValueError("Illegal value for parameter `attack_model_type`.")

    def _build_scaler(self) -> Any | None:
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        if not self.scaler_type:
            return None
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError("Illegal scaler_type: ", self.scaler_type)

    def _build_pipeline(self) -> Pipeline:
        from sklearn.pipeline import Pipeline

        steps: list[tuple[str, Any]] = []
        scaler = self._build_scaler()
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("clf", self._build_base_model()))
        return Pipeline(steps)

    def _search_hyperparameters(
        self, pipeline: Pipeline, x: np.ndarray, y: np.ndarray
    ) -> Pipeline:
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

        distributions = self.param_distributions or _default_param_distributions(
            self.attack_model_type
        )
        cv = StratifiedKFold(
            n_splits=self.search_cv, shuffle=True, random_state=self.random_state
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=distributions,
            n_iter=self.search_n_iter,
            scoring=self.search_scoring,
            cv=cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        search.fit(x, y)
        self.search_results_ = search
        logger.info(
            f"Attack model hyperparameter search ({self.attack_model_type}, "
            f"scoring={self.search_scoring}): best_score={search.best_score_:.4f}, "
            f"best_params={search.best_params_}"
        )
        return search.best_estimator_

    def fit(self, *, members_x: np.ndarray, non_members_x: np.ndarray) -> None:
        """Train the attack model on member (``members_x``) vs non-member (``non_members_x``) features.

        Note on reproducibility with `n_estimators > 1`: base classifiers are
        left with their own `random_state` unseeded so that each bagged copy
        gets genuinely different internal randomness (weight init for "nn",
        split choices for tree models, etc.), in addition to differing via
        bootstrap resampling. `random_state` on this class controls
        BaggingClassifier's own resampling and the hyperparameter search --
        for full end-to-end reproducibility of the *ensemble's* internal
        randomness too, also fix `np.random.seed(...)` before calling `fit`.
        """
        features = members_x.astype(np.float32)
        non_member_features = non_members_x.astype(np.float32)

        labels = np.ones(len(members_x))
        non_member_labels = np.zeros(len(non_members_x))

        x = np.concatenate((features, non_member_features))
        y = np.concatenate((labels, non_member_labels)).astype(np.int64)

        pipeline = self._build_pipeline()

        if self.tune_hyperparameters:
            pipeline = self._search_hyperparameters(pipeline, x, y)

        if self.n_estimators > 1:
            from sklearn.ensemble import BaggingClassifier

            self.model = BaggingClassifier(
                estimator=pipeline,
                n_estimators=self.n_estimators,
                bootstrap=self.bootstrap,
                max_samples=self.max_samples,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            print("self.model =", self.model)
        else:
            self.model = pipeline

        self.model.fit(x, y.ravel())

    def infer(self, *, x: np.ndarray, probabilities: bool = False) -> np.ndarray:
        """Infer membership status (or probabilities) for feature vectors ``x``."""
        if self.model is None:
            raise ValueError("Attack model is not fitted. Call `fit` first.")

        x = x.astype(np.float32)
        inferred = self.model.predict_proba(x)
        return inferred[:, [1]] if probabilities else np.round(inferred[:, [1]])

    @property
    def estimators_(self) -> list[Any] | None:
        """Fitted base pipelines, one per ensemble member (only set when n_estimators > 1)."""
        return getattr(self.model, "estimators_", None)


class MembershipInferenceAttack:
    """Runs a ``MembershipInferenceBlackBox`` end-to-end and reports results.

    Not part of ART -- this is the class ``ModelAttacker`` calls directly.

    New optional ``AttackConfig`` fields this now reads (all via `getattr`
    with the original single-model behavior as the default, so this doesn't
    break until you actually add them to `AttackConfig`):
        n_estimators: int = 1
        bootstrap: bool = False
        max_samples: float = 1.0
        max_features: float = 1.0
        random_state: int | None = None
        n_jobs: int | None = -1
        tune_hyperparameters: bool = False
        param_distributions: dict | None = None
        search_n_iter: int = 25
        search_cv: int = 5
        search_scoring: str = "roc_auc"
    """

    def __init__(self, cfg: AttackConfig) -> None:
        self._cfg = cfg
        self._attack = MembershipInferenceBlackBox(
            attack_model_type=cfg.attack_model_type,
            n_estimators=getattr(cfg, "n_estimators", 10),
            bootstrap=getattr(cfg, "bootstrap", True),
            max_samples=getattr(cfg, "max_samples", 1.0),
            max_features=getattr(cfg, "max_features", 1.0),
            random_state=getattr(cfg, "random_state", None),
            n_jobs=getattr(cfg, "n_jobs", -1),
            tune_hyperparameters=getattr(cfg, "tune_hyperparameters", False),
            param_distributions=getattr(cfg, "param_distributions", None),
            search_n_iter=getattr(cfg, "search_n_iter", 250),
            search_cv=getattr(cfg, "search_cv", 5),
            search_scoring=getattr(cfg, "search_scoring", "roc_auc"),
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

        from art.metrics.privacy.worst_case_mia_score import get_roc_for_fpr

        # get_roc_for_fpr recomputes its own ROC internally from attack_proba /
        # attack_true (rather than reusing roc_fpr/roc_tpr/roc_thresholds above),
        # and picks the operating point closest to -- without exceeding --
        # targeted_fpr. Without `target_model_labels` it returns a single
        # (fpr, tpr, threshold) result; we don't have per-sample target-model
        # class labels in this features-only attack, so no per-class breakdown.
        achieved_fpr, achieved_tpr, threshold = get_roc_for_fpr(
            attack_proba=attack_proba,
            attack_true=attack_true,
            targeted_fpr=self._cfg.targeted_fpr,
        )[0]
        report["worst_case"] = {
            "targeted_fpr": self._cfg.targeted_fpr,
            "tpr": achieved_tpr,
            "fpr": achieved_fpr,
            "threshold": threshold,
        }

        if self._attack.search_results_ is not None:
            report["search_best_params"] = self._attack.search_results_.best_params_
            report["search_best_score"] = float(
                self._attack.search_results_.best_score_
            )

        logger.info(
            f"Membership inference attack ({self._attack.attack_model_type}): {report}"
        )
        return report
