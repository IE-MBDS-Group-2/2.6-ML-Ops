"""Market inputs using past prices, and an unfitted training-only feature recipe."""
import logging

import numpy as np
import pandas as pd
from sklearn import compose, impute, pipeline, preprocessing

TRADING_DAYS_PER_WEEK = 5     # market calendar, not a setting: weeks -> trading days
TRADING_DAYS_PER_YEAR = 252   # market calendar, not a setting: daily volatility -> annual

logger = logging.getLogger(__name__)


def _validate_prices(close_wide: pd.DataFrame) -> None:
    """Reject ambiguous dates, ticker labels and invalid observed prices.

    Inputs: close_wide, the date x ticker price table.
    Outputs: None.
    Why: Features computed on unsorted dates, duplicate tickers or non-positive prices would be silently wrong.
    Raises: ValueError from the checks it calls, or if the table is empty.
    """
    if close_wide.empty:
        raise ValueError("close_wide must contain price rows and ticker columns.")
    _validate_dates(close_wide.index)
    _validate_columns(list(close_wide.columns), "close_wide columns")
    _validate_price_values(close_wide)


def _validate_price_values(close_wide: pd.DataFrame) -> None:
    """Allow missing prices, but reject nonnumeric or invalid observed values.

    Inputs: close_wide, the date x ticker price table.
    Outputs: None.
    Why: Gaps are legitimate (a fund not yet listed), but text, zero, negative or infinite prices are not.
    Raises: ValueError for non-numeric columns, or observed prices that are infinite or not above zero.
    """
    if not all(_is_numeric_price_dtype(dtype) for dtype in close_wide.dtypes):
        raise ValueError(
            "close_wide must contain numeric prices; convert text prices before calling."
        )
    values = close_wide.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or (values <= 0).any():
        raise ValueError(
            "Observed prices must be finite and positive; correct invalid prices upstream."
        )


def _is_numeric_price_dtype(dtype: object) -> bool:
    """Boolean values are not prices, even though Python treats them as numbers.

    Inputs: dtype, a column dtype.
    Outputs: True for numeric, non-boolean dtypes.
    Why: A True / False column would pass a plain numeric check and then be read as the prices 1 and 0.
    """
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def _validate_dates(dates: pd.Index) -> None:
    """Require sorted, unique and nonmissing observation dates.

    Inputs: dates, the index of the price table.
    Outputs: None.
    Why: Trailing windows and shifts count rows, so they only mean "days back" when the dates are in order.
    Raises: ValueError if the index is not a DatetimeIndex with unique, nonmissing, ascending dates.
    """
    if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique:
        raise ValueError("Use a DatetimeIndex with unique, nonmissing dates for close_wide.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Sort close_wide by date before computing features.")


def _validate_columns(columns: list[str], label: str) -> None:
    """Require a nonempty list of distinct, nonblank column names.

    Inputs: columns, the names to check; label, what they are (used in the message).
    Outputs: None.
    Why: Feature names are built from ticker names, so blank or repeated names would clash or hide a column.
    Raises: ValueError if the list is empty, has a non-string or blank name, or repeats a name.
    """
    if not columns or any(not isinstance(name, str) or not name.strip() for name in columns):
        raise ValueError(f"{label} must contain nonblank string column names.")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{label} must not contain duplicate column names.")


def _is_positive_integer(value: int) -> bool:
    """Reject booleans and fractional values as observation windows.

    Inputs: value, a window length from the config.
    Outputs: True if value is an integer above zero and not a boolean.
    Why: A window of 2.5 weeks or True would otherwise reach the shift and rolling calls and misbehave.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_return_weeks(return_weeks: list[int]) -> None:
    """Require distinct positive integer return windows.

    Inputs: return_weeks, the return windows in weeks (features.return_weeks).
    Outputs: None.
    Why: Each window becomes a column name, so a repeat would overwrite a column.
    Raises: ValueError if the list is empty, holds a non-positive or non-integer window, or repeats one.
    """
    if not return_weeks or not all(_is_positive_integer(window) for window in return_weeks):
        raise ValueError("return_weeks must contain positive integer windows, such as [4, 12].")
    if len(set(return_weeks)) != len(return_weeks):
        raise ValueError("return_weeks must not contain duplicate windows.")


def _validate_vol_days(vol_days: int) -> None:
    """Sample standard deviation requires at least two daily returns.

    Inputs: vol_days, the volatility window in trading days (features.vol_days).
    Outputs: None.
    Why: A sample standard deviation (ddof=1) of fewer than two returns is undefined.
    Raises: ValueError if vol_days is not an integer of at least 2.
    """
    if not _is_positive_integer(vol_days) or vol_days < 2:
        raise ValueError("vol_days must be an integer of at least 2 for sample standard deviation.")


def compute_market_features(
    close_wide: pd.DataFrame, return_weeks: list[int], vol_days: int
) -> pd.DataFrame:
    """Compute trailing simple returns and annualized sample volatility without future data.

    Inputs: sorted, unique date-indexed positive prices by ticker; positive return weeks;
    volatility window >= 2. Missing prices are retained, never filled.
    Outputs: same index, ticker-order columns ret_<n>w_<T> then vol_<days>d_<T>.
    Why: expose historical market conditions without look-ahead. A week is five trading
    observations; volatility uses daily simple returns, ddof=1 and sqrt(252) annualization.
    Warm-up and incomplete windows remain NaN. Invalid inputs raise ValueError.
    """
    _validate_prices(close_wide)
    _validate_return_weeks(return_weeks)
    _validate_vol_days(vol_days)
    result = {}
    for ticker in close_wide.columns:
        prices = close_wide[ticker]
        for weeks in return_weeks:
            result[f"ret_{weeks}w_{ticker}"] = prices / prices.shift(TRADING_DAYS_PER_WEEK * weeks) - 1
        daily = prices.pct_change(fill_method=None)
        volatility = daily.rolling(vol_days, min_periods=vol_days).std(ddof=1)
        result[f"vol_{vol_days}d_{ticker}"] = volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    output = pd.DataFrame(result, index=close_wide.index)
    logger.info("Computed %d feature columns for %d price rows", len(output.columns), len(output))
    return output


def dates_with_every_feature(market: pd.DataFrame) -> pd.DatetimeIndex:
    """The dates on which every market feature has a value.

    Inputs:  market, the market features indexed by date (from compute_market_features).
    Outputs: the dates with no missing feature.
    Why:     the longest lookback (the 12-week return, the 20-day volatility) needs history, so the first days
             of any price file have gaps; decisions can only start where every feature exists.
    """
    return market.dropna().index


def get_feature_preprocessor(numeric_columns: list[str]) -> compose.ColumnTransformer:
    """Build an unfitted numeric imputation and scaling recipe.

    Inputs: distinct nonblank numeric column names; invalid lists raise ValueError.
    Outputs: ColumnTransformer with median imputation followed by StandardScaler,
    retaining all-missing columns as zeros and dropping unselected columns.
    Why: train.py fits the recipe on training data only; validation and inference reuse
    it through transform, preventing leakage and preserving feature width.
    """
    _validate_columns(numeric_columns, "numeric_columns")
    numeric = pipeline.Pipeline([
        ("imputer", impute.SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", preprocessing.StandardScaler()),
    ])
    recipe = compose.ColumnTransformer(
        [("numeric", numeric, list(numeric_columns))], remainder="drop"
    )
    logger.info("Created unfitted preprocessor for %d numeric columns", len(numeric_columns))
    return recipe
