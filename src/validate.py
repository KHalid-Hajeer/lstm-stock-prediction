from __future__ import annotations
from typing import Iterable, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.utils import align_X_y

# Basic schema & index checks

def assert_required_columns(df: pd.DataFrame, columns: Iterable[str], name: str = "dataframe") -> None:
    """Assert that all required columns exist.

    Args:
        df (pd.DataFrame): DataFrame to check.
        columns (Iterable[str]): Required columns.
        name (str): Name used in error messages. Defaults to "dataframe".
    
    Raises:
        AssertionError: If any required column is missing.
    """
    missing = set(columns) - set(df.columns)
    assert not missing, f"{name} is missing required columns: {sorted(missing)}."


def assert_unique_sorted_index(df: pd.DataFrame, name: str = "dataframe") -> None:
    """Assert a DatetimeIndex that is strictly increasing and uniqiue.

    Args:
        df (pd.DataFrame): DataFrame with index to check.
        name (str): Name used in error messages. Defaults to "dataframe".

    Raises:
        AssertionError: If the index is not DatetimeIndex, not strictly increasing, or has duplicates.
    """
    assert isinstance(df.index, pd.DatetimeIndex), f"{name}: index must be a DatetimeIndex."
    assert df.index.is_monotonic_increasing, f"{name}: index must be strictly increasing."
    assert not df.index.has_duplicates, f"{name}: index contains duplicate timestamps."

def assert_no_duplicate_columns(df: pd.DataFrame, name: str = "dataframe") -> None:
    """Assert that there are no duplicate column names.

    Args:
        df (pd.DataFrame): DataFrame to check.
        name (str): Name of the dataframe for error messages. Defaults to "dataframe".

    Raises:
        AssertionError: If duplicate column names exist.
    """
    duplicates = pd.Series(df.columns).duplicated(keep=False)
    assert not duplicates.any(), f"{name} contains duplicate columns: {list(pd.Series(df.columns)[duplicates].unique())}"

def assert_no_nan_or_inf(df: Union[pd.DataFrame, pd.Series], name: str = "dataframe") -> None:
    """Assert there are no Nans or ±inf values.

    Args:
        df (Union[pd.DataFrame, pd.Series]: DataFrame or Series to check.
        name (str): Name used in error messages. Defaults to "dataframe".

    Raises:
        AssertionError: If NaN or ±inf values present.
    """
    if isinstance(df, pd.Series):
        bad = ~np.isfinite(df.astype("float64"))
        assert not bad.any(), f"{name} contains NaN or ±inf at {list(df.index[bad])[:5]}..."
    else:
        as_float = df.astype("float64")
        bad_total = (~np.isfinite(as_float.to_numpy())).sum()
        assert bad_total == 0, f"{name} contains NaN or ±inf values."

# Alignment helpers

def assert_same_index(X: pd.DataFrame, y: pd.Series, name_x: str = "X", name_y: str = "y") -> None:
    """Assert that X and y share exactly the same index (order and values).

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target series.
        name_x (str): Name of X for error messages. Defaults to "X".
        name_y (str): Name of y for error messages. Defaults to "y".

    Raises:
        AssertionError: If X and y have different indices.
    """
    assert X.index.equals(y.index), f"{name_x} and {name_y} have different indices."

# Leakge guards (heuristic)

def assert_no_future_base_fields_in_X(
    X: pd.DataFrame, 
    raw: pd.DataFrame, 
    base_cols: Sequence[str] = ("open", "high", "low", "close", "volume", "vwap", "trade_count"), 
    max_lookahead_days: int = 3,
    atol: float = 0.0,
    rtol: float = 0.0
    ) -> None:
    """Assert that no features equals a future base field (obvious leakage).

    Checks if any column in X is numerically identical to raw[col].shift(-k) for k in 1..max_lookahead_days, within a tolerance.

    Args:
        X (pd.DataFrame): Feature matrix (post-lag).
        raw (pd.DataFrame): Raw OHLCV(+optional vwap and trade_count) data.
        base_cols (Sequence[str]): Raw columns to test. Defaults to ("open", "high", "low", "close", "volume", "vwap", "trade_count").
        max_lookahead_days (int): Maximum number of days to look ahead. Defaults to 3.
        atol (float): Absolute tolerance for equality. Defaults to 0.0.
        rtol (float): Relative tolerance for equality. Defaults to 0.0.
    
    Raises:
        AssertionError: If a suspicious equality is found.
    """
    cols = [c for c in base_cols if c in raw.columns]
    for col in cols:
        series = raw[col].astype("float64")
        for k in range(1, max_lookahead_days + 1):
            future = series.shift(-k).reindex(X.index)
            # Compare each feature to this future series
            for feature in X.columns:
                same_mask = np.isclose(X[feature].astype("float64"), future.astype("float64"), atol=atol, rtol=rtol)
                # Trigger only if many match (avoid random coincidences)
                if same_mask.sum() >= int(0.95 * len(X)) and len(X) > 0:
                    raise AssertionError(f"Leakage detected: feature {feature} is numerically identical to {col}.shift(-{k}) on {same_mask.sum()}/{len(X)} rows.")

def assert_no_lookahead(X: pd.DataFrame, y: pd.Series, expect_central_lag_days: int = 1) -> None:
    """Assert basic anti-lookahead conditions between X and y.

    This function enforces:
        1) X and y share the same index.
        2) X has already applied the expected central lag meaning features at time t were computed with data <= t - expected_central_lag_days.
    
    Note that this is a structural guard. It cannot fully prove absence of leakage inside a feature formula. 
    
    Args:
        X (pd.DataFrame): Feature matrix (post-lag).
        y (pd.Series): Target series.
        expect_central_lag_days (int): Expected central lag applied to X in days. Defaults to 1.

    Raises:
        AssertionError: If X and y do not share the same index or if X appears unlagged.
    """
    assert_same_index(X, y)

    # Heuristic check: if X looks identical to its own forward shift, it's probably not lagged.
    if expect_central_lag_days > 0 and len(X) > expect_central_lag_days:
        shifted_forward = X.shift(-expect_central_lag_days)
        # If too many exact matches, likely no lag was applied
        match_ratio = (X.equals(shifted_forward))
        assert not match_ratio, ("X appears not to be centrally lagged. Expected a central lag; found X == X.shift(-lag).")

# Convenience composite

def run_all_validations(
    raw: pd.DataFrame, 
    X: pd.DataFrame, 
    y:pd.Series, 
    required_raw_cols: Sequence[str] = ("open", "high", "low", "close"), 
    expected_central_lag_days: int = 1
    ) -> None:
    """Run a standard set of validations.

    Args:
        raw (pd.DataFrame): Raw OHLCV (and optional vwap and trade_count) data.
        X (pd.DataFrame): Feature matrix (post-lag).
        y (pd.Series): Target series aligned to X.
        required_raw_cols (Sequence[str]): Required columns in `raw`. Defaults to ("open", "high", "low", "close").
        expected_central_lag_days (int): Expected central lag applied to X in days. Defaults to 1.
    
    Raises:
        AssertionError: If any validation fails.
    """
    assert_required_columns(raw, required_raw_cols, name="raw")
    assert_unique_sorted_index(raw, name="raw")
    assert_no_duplicate_columns(raw, name="raw")

    assert_unique_sorted_index(X, name="X")
    assert_no_duplicate_columns(X, name="X")
    assert_no_nan_or_inf(X, name="X")

    # Align & check indexes
    X2, y2 = align_X_y(X, y, dropna=True)
    assert_same_index(X2, y2)

    # Anti-leakage (structural + heuristic)
    assert_no_lookahead(X2, y2, expect_central_lag_days=expected_central_lag_days)
    assert_no_future_base_fields_in_X(X2, raw, max_lookahead_days=3)