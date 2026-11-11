"""
train.py  [Step 5: TRAIN]
=========================
Purpose   Bundle the feature recipe + the algorithm into ONE pipeline and fit it
Input     X_train, y_train + unfitted recipe from features.py + model settings
Output    One fitted pipeline (preprocessing + model), saved by main.py as models/model.joblib
Called by main.py          Next   evaluate.py

Why it matters
- .fit() runs on the train split only, so validation and test metrics stay honest
- One artifact for preprocessing + model: what you test is exactly what you deploy
- Model choice and hyperparameters come from config: swap models without rewriting code
- Fixed random seed: same data, same result

Tests     tests/test_train.py (stage 2)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

SUPPORTED_PROBLEM_TYPES = ("classification",)


# Subclassing LogisticRegression inherits sklearn's own deep class tree, hence the ancestors waiver.
class ActionClassifier(LogisticRegression):  # pylint: disable=too-many-ancestors
    """LogisticRegression that accepts the fractional action labels 0.0, 0.5 and 1.0.

    Why: sklearn reads float labels such as 0.5 as a continuous (regression) target and refuses to fit a
    classifier. We fit on class codes 0, 1, 2 and put the real labels back in classes_, so predict returns
    0.0 / 0.5 / 1.0 and predict_proba columns follow classes_. Same parameters as LogisticRegression.
    """

    def fit(self, X, y, sample_weight=None):
        labels, codes = np.unique(np.asarray(y), return_inverse=True)
        super().fit(X, codes, sample_weight=sample_weight)
        # sklearn convention: fitted attributes (trailing _) are set in fit, never in __init__.
        self.classes_ = labels  # pylint: disable=attribute-defined-outside-init
        return self


def _check_inputs(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str, model_params: dict) -> None:
    """Fail fast before fitting: wrong problem type, no rows, X and y of different lengths, seed given twice."""
    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise ValueError(f"problem_type must be one of {SUPPORTED_PROBLEM_TYPES}, got {problem_type!r}. "
                         "Check problem.problem_type in config.yaml.")
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train and y_train must not be empty. Check the split dates in config.yaml.")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train must have the same number of rows, got {len(X_train)} "
                         f"and {len(y_train)}.")
    if "random_state" in model_params:
        raise ValueError("Pass random_state as the argument (model.random_state in config.yaml), "
                         "not inside model_params.")


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer, problem_type: str,
                model_params: dict | None = None, random_state: int = 42) -> Pipeline:
    """Course contract: bundle the feature recipe and LogisticRegression into one pipeline and fit it.

    Inputs:  X_train, y_train (best_action: 0.0, 0.5, 1.0) from the train split only; preprocessor, the
             UNFITTED recipe from features.get_feature_preprocessor; problem_type ("classification");
             model_params from config.yaml model.params (e.g. max_iter, class_weight); random_state, the seed.
    Outputs: the fitted Pipeline([("preprocess", preprocessor), ("model", LogisticRegression)]).
    Why:     the only .fit() in the project, on train rows only, so validation and test stay honest;
             one artifact means what we evaluate is exactly what infer loads.
    Raises:  ValueError for an unsupported problem_type, empty data, X and y of different lengths, or
             random_state inside model_params.
    """
    params = dict(model_params or {})
    _check_inputs(X_train, y_train, problem_type, params)
    counts = y_train.value_counts().sort_index()
    logger.info("Training on %d rows; class counts %s",
                len(X_train), ", ".join(f"{label}: {count}" for label, count in counts.items()))
    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", ActionClassifier(random_state=random_state, **params)),
    ])
    model.fit(X_train, y_train)
    logger.info("Fitted %s with %d features", type(model.named_steps["model"]).__name__, X_train.shape[1])
    return model
