"""
infer.py  [Step 7: INFER]
=========================
Purpose   Turn new, unseen weeks into Friday's recommendation with the saved pipeline, falling back to today's
          rule when the model is unsure or an input is missing
Input     Fitted pipeline + new data (DataFrame, same feature columns, no target)
Output    DataFrame with a fixed schema: "prediction" (+ optional "probability"), same index as input;
          apply_fallback adds "recommendation" and "fallback"
Called by main.py (batch, writes reports/predictions.csv) and api.py (stage 3, live requests)

Why it matters
- No training here: inference is reused unchanged by the API, so batch and live results match
- Fixed output columns: downstream teams can rely on them
- Keep the input index (the Friday date) so predictions join back to source records
- The fallback means a shaky model can never do worse than today's rule on the weeks it is unsure about

Tests     tests/test_infer.py
"""

import logging

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def run_inference(model: Pipeline, X_infer: pd.DataFrame, include_proba: bool = False) -> pd.DataFrame:
    """Predict the action for each new row with a fitted pipeline.

    Inputs:  model: fitted pipeline; X_infer: feature rows (no target), indexed by date or ID;
             include_proba: also return the model's confidence in its chosen action.
    Outputs: DataFrame with the same index as X_infer, column "prediction" and, if asked, "probability"
             (the highest class probability per row). X_infer is not modified.
    Why:     One fixed output shape for batch runs and the stage 3 API; probability feeds apply_fallback.
    """
    if len(X_infer) == 0:
        raise ValueError("run_inference: X_infer is empty. Check the inference data was loaded and filtered correctly.")
    preds = pd.DataFrame({"prediction": model.predict(X_infer)}, index=X_infer.index)
    if include_proba:
        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "run_inference: include_proba=True but this model has no predict_proba. "
                "Use a classifier that outputs probabilities (e.g. LogisticRegression, RandomForest)."
            )
        preds["probability"] = model.predict_proba(X_infer).max(axis=1)
    logger.info("Predicted %d rows", len(preds))
    return preds


def apply_fallback(preds: pd.DataFrame, rule_actions: pd.Series, missing_input: pd.Series,
                   min_probability: float = 0.60) -> pd.DataFrame:
    """Replace unsure or incomplete predictions with today's rule action.

    Inputs:  preds: output of run_inference(..., include_proba=True); rule_actions: today's rule action per row;
             missing_input: True where any model input was missing; min_probability: confidence needed (default 0.60).
             All three share the same index.
    Outputs: Copy of preds plus "recommendation" (the action to take) and "fallback" (True where the rule was used).
    Why:     The model only overrides today's rule when it is confident and fully informed.
    """
    if "probability" not in preds.columns:
        raise ValueError(
            "apply_fallback: preds has no 'probability' column. Call run_inference with include_proba=True."
        )
    if not (rule_actions.index.equals(preds.index) and missing_input.index.equals(preds.index)):
        raise ValueError(
            "apply_fallback: preds, rule_actions and missing_input must have the same index (same dates, same order)."
        )
    out = preds.copy()
    out["fallback"] = (out["probability"] < min_probability) | missing_input.astype(bool)
    out["recommendation"] = out["prediction"].where(~out["fallback"], rule_actions)
    out = out[["prediction", "probability", "recommendation", "fallback"]]
    logger.info("Fallback to today's rule on %d of %d rows", int(out["fallback"].sum()), len(out))
    return out
