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
