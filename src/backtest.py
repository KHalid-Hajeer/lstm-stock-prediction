from __future__ import annotations

from typing import Literal, Optional
import numpy as np
import pandas as pd
from src.utils import align_X_y

EPSILON: float = 1e-12

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a time-series z-score with rolling mean and std (no lookahead).

    Args:
        series(pd.Series): Input score series (time-sorted).
        window(int): Rolling window length for mean and std.

    Returns:
        pd.Series: Z-scored series; first window-1 values are NaN.
    """
    series = pd.Series(series).astype("float64")
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / (std + EPSILON)

def _ewma_std(series: pd.Series, span: int) -> pd.Series:
    """Compute Exponentially Weighted Moving Average (EWMA) standard deviation (no lookahead).

    Args:
        series(pd.Series): Input score series (time-sorted).
        span(int): EWMA span (days).

    Returns: 
        pd.Series: EWMA std; first span-1 values are NaN.
    """
    series = pd.Series(series).astype("float64")
    return series.ewm(span=span, adjust=False, min_periods=span).std()

def single_asset_backtest(
    scores: pd.Series,
    y_raw_next: pd.Series,
    z_window: int = 60,
    raw_weight_cap: float = 1.0,
    target_vol_annual: float = 0.10,
    vol_span: int = 20,
    cost_bps: float = 5.0,
    gross_limit: float = 3.0,
    execution: Literal["close_to_close", "close_to_open"] = "close_to_close",
    drop_warmup: bool = True,
    use_vol_targeting: bool = True,
    ) -> pd.DataFrame:
    """Convert daily scores into positions and after-cost PnL for a single asset.

    All computations at time t use information available up to and including t.

    Args:
        scores (pd.Series): Model scores at time t intended to predict r_{t+1}.
        y_raw_next (pd.Series): Next-day raw return r_{t+1} aligned to 'scores'.
        z_window (int, optional): Lookback for time-series z-scoring of scores. Defaults to 60. 
        raw_weight_cap (float, optional): Cap on |pre-scaling weight|. Defaults to 1.0
        target_vol_annual (float, optional): Target annualised volatility.Used only if 'use_vol_targeting' is True. Defaults to 0.10.
        vol_span (int, optional): EWMA span (days) for estimating asset volatility from r_t. Defaults to 20.
        cost_bps (float, optional): one-way transaction cost in basis points per unit turnover (|Δw|)). Defaults to 5.0.
        gross_limit (float, optional): Hard cap on |final weight| after risk scaling. Defaults to 3.0.
        execution (Literal["close_to_close", "close_to_open"], optional): Return convention for 'y_raw_next' ("close_to_open" or "close_to_close"). Doc flag only; not used in computation. Defaults to "close_to_close".
        drop_warmup (bool, optional): Drop rows where z-score/weights not yet defined. Defaults to True.
        use_vol_targeting (bool, optional): If False, skip risk scaling (multiplier = 1). Defaults to True.

    Returns:
        pd.DataFrame: Index = dates (aligned). Columns:
            score                  : input scores at t
            z_score                : rolling z-scored signal
            weight_pre_scaling     : capped pre-risk-scaling weight
            asset_volatility       : EWMA vol of r_t (computed with data ≤ t)
            risk_scaling_multiplier: daily multiplier to hit target vol (or 1 if disabled)
            weight                 : final weight after scaling and gross cap
            turnover               : |w_t - w_{t-1}|
            gross_return           : w_t * r_{t+1}
            transaction_cost       : turnover * cost_bps / 10,000
            net_return             : gross_return - transaction_cost

    Raises:
        ValueError: On invalid parameter settings or if indices do not overlap.
    """
    # Parameter validation
    if z_window < 5:
        raise ValueError("z_window must be at least 5.")
    if raw_weight_cap < 0:
        raise ValueError("raw_weight_cap must be non-negative.")
    if gross_limit < raw_weight_cap:
        raise ValueError("gross_limit must be >= raw_weight_cap.")
    if vol_span < 5:
        raise ValueError("vol_span must be at least 5.")
    if cost_bps < 0.0:
        raise ValueError("cost_bps must be non-negative.")
    if use_vol_targeting and target_vol_annual <= 0.0:
        raise ValueError("target_vol_annual must be positive when use_vol_targeting is True.")
    if execution not in ("close_to_close", "close_to_open"):
        raise ValueError("execution must be 'close_to_close' or 'close_to_open'.")
    
    # Align inputs and sanity checks
    scores_series = pd.Series(scores).astype("float64")
    next_day_return = pd.Series(y_raw_next).astype("float64")

    X = pd.DataFrame({"score": scores_series})
    y = next_day_return.copy()
    try:
        X_aligned, y_aligned = align_X_y(X, y, dropna=False)
    except ValueError:
        # Fallback: adopt y's DatetimeIndex if lengths match; else align by position with RangeIndex.
        if len(X) != len(y):
            raise 
        if isinstance(y.index, pd.DatetimeIndex):
            X.index = y.index
        else:
            # If neither side is datetime, align by position without inventing dates
            rng = pd.RangeIndex(len(y))
            X.index = rng
            y.index = rng
        X_aligned, y_aligned = align_X_y(X, y, dropna=False)
    
    scores_series = X_aligned["score"]
    next_day_return = y_aligned

    # 1) Time-series z-score of model scores (no lookahead)
    z_score = _rolling_zscore(scores_series, z_window)

    # 2) Raw (pre-scaling) weight with cap
    weight_pre_scaling = z_score.clip(-raw_weight_cap, raw_weight_cap)

    # 3) Risk scaling (vol targeting) using yesterday's realised return r_t
    realized_return_t = next_day_return.shift(1) # r_t from history (no lookahead)
    asset_volatility = _ewma_std(realized_return_t, vol_span)

    if use_vol_targeting:
        daily_target_vol = target_vol_annual / np.sqrt(252.0)
        # Limit scaling so final |weight| <= gross_limit
        max_multiplier = gross_limit / max(raw_weight_cap, EPSILON)
        risk_scaling_multiplier = (daily_target_vol / (asset_volatility + EPSILON)).clip(upper=max_multiplier)
    else:
        risk_scaling_multiplier = pd.Series(1.0, index=scores_series.index)
    
    # Final weight, capped by gross_limit
    weight = (weight_pre_scaling * risk_scaling_multiplier).clip(-gross_limit, gross_limit)

    # 4) Turnover & costs
    turnover = (weight - weight.shift(1)).abs().fillna(0.0)
    transaction_cost = turnover * (cost_bps / 10_000.0)

    # 5) PnL
    gross_return = weight * next_day_return
    net_return = gross_return - transaction_cost

    backtest_frame = pd.DataFrame(
        {
            "score": scores_series,
            "z_score": z_score,
            "weight_pre_scaling": weight_pre_scaling,
            "asset_volatility": asset_volatility,
            "risk_scaling_multiplier": risk_scaling_multiplier,
            "weight": weight,
            "turnover": turnover,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return
        }
    )

    if drop_warmup:
        cols_to_check = ["z_score", "weight"]
        backtest_frame = backtest_frame.dropna(subset=cols_to_check)

    backtest_frame.index.name = "date" if isinstance(backtest_frame.index, pd.DatetimeIndex) else "idx"
    return backtest_frame