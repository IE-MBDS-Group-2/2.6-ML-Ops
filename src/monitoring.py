"""
monitoring.py  [Stage 3: IS THE MODEL STILL FIT TO USE?]
========================================================
Purpose   Two numbers the retrain triggers use: how far the market inputs have drifted from the train years
          (population stability index, PSI) and how often the live service falls back to today's rule
Input     Daily returns (date x ticker) for the train years and for recent days; the answers the API logged to
          W&B (as_of, fallback)
Output    PSI per ticker, the drift per week, the fallback rate per week, and whether each alert fires
Used by   scripts/monitor_report.py (the weekly monitoring run in W&B)

Why it matters
- Open gap 12: the drift alert had no number. PSI is the standard credit-scoring drift measure, with the usual
  reading below 0.1 stable, 0.1 to 0.2 watch, above 0.2 a real shift; the alert fires above
  monitoring.psi_limit (0.2)
- Measured on daily returns, the input every market feature is built from, not on the features themselves: the
  4- and 12-week returns and the 20-day volatility overlap from one day to the next, so a few months hold only a
  handful of independent values and their PSI is far above 0.2 even when nothing has changed (on the synthetic
  prices, a median of 3.5 over 120 days). Daily returns are close to independent, so the noise level is known
- Bins are deciles of the train years, so each bin holds a tenth of the reference rows
- Pure functions on DataFrames: no W&B and no files here, so every number is tested on small known tables

Tests     tests/test_monitoring.py
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SHARE_FLOOR = 1e-4   # an empty bin counts as this share, so log(0) never happens (the usual PSI convention)
FRIDAY = 4           # pandas day_of_week: Monday is 0


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int) -> float:
    """PSI of one feature: sum over bins of (current share - reference share) x ln(current / reference share).

    Inputs:  reference, the feature over the train years; current, the same feature over recent days;
             bins, the number of quantile bins cut on the reference (10 = deciles).
    Outputs: the PSI, 0.0 when both share the same distribution, larger the more they differ.
    Why:     bin edges come from the reference only, so the recent data never moves the yardstick. Missing
             values (feature warm-up) are dropped. Repeated values can merge quantile edges: the bins are then
             fewer, never empty by construction.
    Raises:  ValueError if bins < 2 or either side has no values.
    """
    if bins < 2:
        raise ValueError(f"population_stability_index: bins must be at least 2, got {bins}.")
    reference, current = reference.dropna(), current.dropna()
    if reference.empty or current.empty:
        raise ValueError("population_stability_index: reference and current both need at least one value. "
                         "Check the feature warm-up and the dates of each period.")
    inner_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    reference_share = _bin_shares(reference, edges)
    current_share = _bin_shares(current, edges)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def _bin_shares(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Share of values in each bin (a value on an edge goes to the upper bin), empty bins raised to SHARE_FLOOR."""
    counts, _ = np.histogram(values, bins=edges)
    return np.maximum(counts / counts.sum(), SHARE_FLOOR)


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], bins: int) -> pd.Series:
    """PSI of each column (a ticker's daily return, or any feature), largest first.

    Inputs:  reference and current, date x column tables; columns, the ones to compare; bins, as above.
    Outputs: Series named "psi", indexed by column, sorted from the largest drift down.
    Raises:  KeyError naming any column missing from either table.
    """
    missing = [column for column in columns if column not in reference or column not in current]
    if missing:
        raise KeyError(f"feature_drift: columns missing from the reference or current table: {missing}")
    psi = pd.Series({column: population_stability_index(reference[column], current[column], bins)
                     for column in columns}, name="psi")
    return psi.sort_values(ascending=False)


def weekly_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], bins: int,
                 window_days: int) -> pd.DataFrame:
    """The drift measure on each Friday: the largest PSI over the columns, on the trailing window_days rows.

    Inputs:  reference, current and columns as in feature_drift (current indexed by trading date);
             window_days, how many trading days each Friday looks back.
    Outputs: one row per Friday with a full window: columns "max_psi" and "feature" (the one that drifted most).
             Empty when current is shorter than window_days.
    Why:     one number per week, comparable with the weekly fallback rate. The window must be long enough
             for the sample to fill the bins: with n independent rows and B bins, noise alone gives a PSI of
             about (B - 1) / n, 0.05 for 180 days and 10 bins, well under the 0.2 limit.
    Raises:  ValueError if window_days < 1.
    """
    if window_days < 1:
        raise ValueError(f"weekly_drift: window_days must be at least 1, got {window_days}.")
    current = current.dropna(subset=columns)
    fridays = [day for position, day in enumerate(current.index)
               if day.day_of_week == FRIDAY and position + 1 >= window_days]
    rows = []
    for friday in fridays:
        window = current.loc[:friday].tail(window_days)
        psi = feature_drift(reference, window, columns, bins)
        rows.append({"week": friday, "max_psi": psi.iloc[0], "feature": psi.index[0]})
    return pd.DataFrame(rows, columns=["week", "max_psi", "feature"]).set_index("week")


def weekly_fallback_rate(answers: pd.DataFrame, date_column: str, fallback_column: str) -> pd.DataFrame:
    """Share of answers that fell back to today's rule, per week (weeks end on Friday).

    Inputs:  answers, one row per API answer; date_column, the decision date (as_of);
             fallback_column, True when the answer is today's rule.
    Outputs: one row per week ending Friday: "fallback_rate" (0 to 1) and "answers" (how many it is based on).
    Why:     the first retrain trigger is a fallback rate above 50% for 4 weeks in a row.
    Raises:  KeyError naming a missing column; ValueError if there are no answers.
    """
    missing = [column for column in (date_column, fallback_column) if column not in answers]
    if missing:
        raise KeyError(f"weekly_fallback_rate: answers have no columns {missing}. Expected the serve table "
                       "the API logs to W&B.")
    if answers.empty:
        raise ValueError("weekly_fallback_rate: no answers to measure. Check the serve runs in W&B.")
    week = pd.to_datetime(answers[date_column]).dt.to_period("W-FRI").dt.end_time.dt.normalize()
    fallback = answers[fallback_column].astype(str).str.lower().isin(["true", "1", "1.0"])
    grouped = fallback.groupby(week.values)
    table = pd.DataFrame({"fallback_rate": grouped.mean(), "answers": grouped.size()})
    table.index.name = "week"
    return table


def drift_alert(drift: pd.DataFrame, limit: float) -> bool:
    """True when the latest week's drift measure is above the limit."""
    return bool(not drift.empty and drift["max_psi"].iloc[-1] > limit)


def fallback_alert(rates: pd.DataFrame, limit: float, weeks: int) -> bool:
    """True when the fallback rate is above the limit in each of the last `weeks` weeks."""
    latest = rates["fallback_rate"].tail(weeks)
    return bool(len(latest) == weeks and (latest > limit).all())
