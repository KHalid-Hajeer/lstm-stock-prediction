from typing import Dict, Iterable, Optional, Tuple
import pandas as pd
import numpy as np

def clean_series(series: pd.Series) -> pd.Series:
    """Return float64 series with only finite values (index preserved).
    
    Args:
        series (pd.Series): Input series.

    Returns:
        pd.Series: Float64 series with only finite values (original index preserved).

    Notes:
        - We deliberately do NOT raise when the result is empty.
        Returning an empty series allows callers to decide whether to error or skip.
        - Keeps original index for the retained entries only.
    """
    series_float64 = pd.Series(series).astype("float64")
    mask = np.isfinite(series_float64.values)
    return series_float64[mask]

def align_X_y(X: pd.DataFrame, y: pd.Series, dropna: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features (X) and target (y) by index; optionally drop rows with any NaNs.
    
    Args:
        X: Feature matrix.
        y: Target series.
        dropna_rows: Whether to drop rows with any NaNs.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Aligned feature matrix (X) and target vector (y) with identical index.

    Raises:
        ValueError: If there is no overlapping index or the result is empty after cleaning.
    """
    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()

    idx = X.index.intersection(y.index)
    if idx.empty:
        raise ValueError("No overlapping index between X and y.")
    
    X2, y2 = X.loc[idx], y.loc[idx]

    if dropna:
        X_f64 = X2.astype("float64")
        y_f64 = y2.astype("float64")
        mask = (
            np.isfinite(X_f64.to_numpy()).all(axis=1)
            & np.isfinite(y_f64.to_numpy())
        )
        X2, y2 = X2.loc[mask], y2.loc[mask]

        if X2.empty or y2.empty:
            raise ValueError("After alignemnt, X or y are empty. Check NaN/inf values.")
    return X2, y2

def as_series(values: np.ndarray | Iterable[float], index: pd.Index, name: str) -> pd.Series:
    """Convert array-like predictions to a Series with the given index/name.
    
    Args:
        values (np.ndarray | Iterable[float]): Array-like numeric predictions.
        index (pd.Index): Index for the Series.
        name (str): Series name.
    
    
    Returns:
        pd.Series: Series with the given index/name.
    
    Raises:
        ValueError: If lengths do not match.
    """
    array = np.asarray(values, dtype="float64")
    if len(array) != len(index):
        raise ValueError(f"Length mismatch: values={len(array)} vs index={len(index)}")
    return pd.Series(array, index=index, name=name)