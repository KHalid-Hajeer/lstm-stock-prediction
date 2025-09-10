# src/metrics.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.utils import clean_series


# Internal helpers

def infer_periods_per_year(index: pd.Index, default: int = 252) -> int:
    """Infer trading periods per year from a Date/Datetime index.

    Args:
        index: Index to inspect (ideally a DatetimeIndex).
        default: Fallback if frequency is unknown.

    Returns:
        int: Estimated periods per year (e.g., 252 for daily).
    """
    if not isinstance(index, pd.DatetimeIndex) or index.size < 2:
        return default
    freq = pd.infer_freq(index)
    if not freq:
        return default
    f = freq.upper()
    if f.startswith(("B", "D")):
        return 252
    if f.startswith("W"):
        return 52
    if f.startswith(("M", "SM")):
        return 12
    if f.startswith("Q"):
        return 4
    if f.startswith(("A", "Y")):
        return 1
    return default


def risk_free_rate_per_period(rf_annual: float, periods_per_year: int) -> float:
    """Convert annual risk-free rate to per-period using geometric mapping.

    Args:
        rf_annual: Annual risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        float: Per-period risk-free rate.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    return (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0


# Core performance statistics

def annualized_return(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    returns_are_log: bool = False,
) -> float:
    """Compute annualised return (Compound Annual Growth Rate) from a return series.

    Args:
        returns: Per-period returns (simple or log), indexed by date.
        periods_per_year: Override periods per year; inferred if None.
        returns_are_log: If True, interpret `returns` as log returns.

    Returns:
        float: Annualised return (CAGR) as a decimal (e.g., 0.12 for 12%).

    Notes:
        For log returns, we exponentiate the cumulative sum; for simple returns, we compound (1+r). If the input is empty, returns NaN.
    """
    r = clean_series(returns)
    if r.empty:
        return np.nan
    ppy = periods_per_year or infer_periods_per_year(r.index)
    n = r.size
    if returns_are_log:
        total_simple = np.exp(r.sum()) - 1.0
        return (1.0 + total_simple) ** (ppy / n) - 1.0
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return np.nan
    return growth ** (ppy / n) - 1.0


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
) -> float:
    """Compute annualised standard deviation of (arithmetic) returns.

    Args:
        returns: Per-period returns (treated arithmetically).
        periods_per_year: Override periods per year; inferred if None.

    Returns:
        float: Annualised volatility.
    """
    r = clean_series(returns)
    if r.empty:
        return np.nan
    ppy = periods_per_year or infer_periods_per_year(r.index)
    return float(r.std(ddof=0) * np.sqrt(ppy))


def downside_std(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    mar: float = 0.0,
) -> float:
    """Compute annualised downside deviation relative to a Minimum Acceptable Return (MAR).

    Args:
        returns: Per-period returns.
        periods_per_year: Override periods per year; inferred if None.
        mar: Per-period MAR.

    Returns:
        float: Annualised downside deviation.
    """
    r = clean_series(returns)
    if r.empty:
        return np.nan
    ppy = periods_per_year or infer_periods_per_year(r.index)
    downside = np.clip(r - mar, a_max=0.0, a_min=None)
    per_period_downside = float(np.sqrt((downside ** 2).mean()))
    return per_period_downside * np.sqrt(ppy)


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    risk_free_rate_annual: float = 0.0,
) -> float:
    """Compute annualised Sharpe ratio.

    Args:
        returns: Per-period returns (after costs recommended).
        periods_per_year: Override periods per year; inferred if None.
        risk_free_rate_annual: Annual risk-free rate.

    Returns:
        float: Sharpe ratio (unitless). NaN if variance is zero or input empty.
    """
    r = clean_series(returns)
    if r.empty:
        return np.nan
    ppy = periods_per_year or infer_periods_per_year(r.index)
    rf = risk_free_rate_per_period(risk_free_rate_annual, ppy)
    excess = r - rf
    mu = float(excess.mean())
    sigma = float(excess.std(ddof=0))
    if sigma == 0.0:
        return np.nan
    return (mu / sigma) * np.sqrt(ppy)


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    mar: float = 0.0,
) -> float:
    """Compute annualised Sortino ratio (excess over MAR / downside deviation).

    Args:
        returns: Per-period returns (after costs recommended).
        periods_per_year: Override periods per year; inferred if None.
        mar: Per-period minimum acceptable return.

    Returns:
        float: Sortino ratio (unitless). NaN if downside deviation is zero or input empty.
    """
    r = clean_series(returns)
    if r.empty:
        return np.nan
    ppy = periods_per_year or infer_periods_per_year(r.index)
    excess = r - mar
    mu = float(excess.mean())
    dstd_annual = downside_std(r, periods_per_year=ppy, mar=mar)
    if dstd_annual == 0.0 or np.isnan(dstd_annual):
        return np.nan
    # Convert annual downside deviation back to per-period for the ratio
    dstd_per_period = dstd_annual / np.sqrt(ppy)
    return (mu / dstd_per_period) * np.sqrt(ppy)


# Wealth and drawdowns

def wealth_curve(
    returns: pd.Series,
    starting_wealth: float = 1_000.0,
    returns_are_log: bool = False,
) -> pd.Series:
    """Compute wealth curve from returns.

    Args:
        returns: Per-period returns (simple or log).
        starting_wealth: Initial capital; must be positive.
        returns_are_log: If True, interpret returns as log returns.

    Returns:
        pd.Series: Wealth time series.

    Raises:
        ValueError: If starting_wealth <= 0.
    """
    if starting_wealth <= 0:
        raise ValueError("starting_wealth must be positive.")
    r = clean_series(returns)
    if r.empty:
        return pd.Series(dtype="float64")
    growth = np.exp(r.cumsum()) if returns_are_log else (1.0 + r).cumprod()
    wealth = starting_wealth * growth
    wealth.name = "wealth"
    return wealth


def drawdown_series(wealth: pd.Series) -> pd.Series:
    """Compute drawdown series from a wealth curve.

    Args:
        wealth: Wealth time series (positive).

    Returns:
        pd.Series: Drawdown series in [-1, 0].
    """
    w = clean_series(wealth)
    if w.empty:
        return pd.Series(dtype="float64")
    running_peak = w.cummax()
    dd = (w / running_peak) - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(wealth: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Compute maximum drawdown and its peak/trough dates.

    Args:
        wealth: Wealth time series.

    Returns:
        Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
            (max_drawdown, peak_date, trough_date), where drawdown is negative.
    """
    dd = drawdown_series(wealth)
    if dd.empty:
        return (np.nan, None, None)
    trough = dd.idxmin()
    maxdd = float(dd.loc[trough])
    peak = wealth.loc[:trough].idxmax()
    return (maxdd, peak, trough)


# Misc metrics

def hit_rate(returns: pd.Series) -> float:
    """Compute fraction of periods with positive returns.

    Args:
        returns: Per-period returns.

    Returns:
        float: Hit rate in [0, 1] or NaN if input empty.
    """
    r = clean_series(returns)
    if r.empty:
        return np.nan
    return float((r > 0).mean())


def turnover_stats(
    turnover: pd.Series,
    periods_per_year: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute average daily turnover and annualised turnover.

    Args:
        turnover: |w_t - w_{t-1}| series.
        periods_per_year: Override periods per year; inferred if None.

    Returns:
        Tuple[float, float]: (avg_daily_turnover, annualized_turnover), NaNs if empty.
    """
    t = clean_series(turnover)
    if t.empty:
        return (np.nan, np.nan)
    ppy = periods_per_year or infer_periods_per_year(t.index)
    avg = float(t.mean())
    return (avg, avg * ppy)


# Trade-level analytics

def compute_trades(
    weight: pd.Series,
    returns: pd.Series,
    min_abs_weight: float = 1e-6,
) -> pd.DataFrame:
    """Segment a backtest into trades (consecutive, non-zero weight with constant sign).

    Args:
        weight: Strategy weight time series (will be sorted by index if unsorted).
        returns: Per-period returns aligned to `weight` (after costs recommended).
        min_abs_weight: Threshold below which weight is treated as flat (0).

    Returns:
        pd.DataFrame: One row per trade with:
            trade_id, side (+1 long / -1 short), start, end, n_days, pnl, pnl_per_day, hit (0/1).

    Notes:
        - If there are no trades (always flat), returns an empty DataFrame.
        - A "mask" is used internally to keep only time steps meeting a condition.
    """
    w = pd.Series(weight).astype("float64")
    r = pd.Series(returns).astype("float64").reindex(w.index)
    if not w.index.is_monotonic_increasing:
        w = w.sort_index()
        r = r.reindex(w.index)

    side = np.sign(w.where(w.abs() >= min_abs_weight, 0.0))
    new_trade_mask = (side != side.shift(1)) & (side != 0)
    trade_id = new_trade_mask.cumsum() * (side != 0)

    rows = []
    for tid, idx in trade_id[trade_id > 0].groupby(trade_id).groups.items():
        start_dt, end_dt = idx[0], idx[-1]
        segment = r.loc[start_dt:end_dt].dropna()
        if segment.empty:
            continue
        this_side = int(side.loc[start_dt])
        pnl = float(segment.sum())
        n_days = int(segment.size)
        rows.append(
            {
                "trade_id": int(tid),
                "side": this_side,
                "start": start_dt,
                "end": end_dt,
                "n_days": n_days,
                "pnl": pnl,
                "pnl_per_day": pnl / n_days if n_days > 0 else np.nan,
                "hit": 1 if pnl > 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def trade_level_summary(
    weight: pd.Series,
    returns: pd.Series,
    min_abs_weight: float = 1e-6,
) -> pd.Series:
    """Summarise trade-level performance.

    Args:
        weight: Strategy weight time series.
        returns: Per-period returns aligned to `weight` (after costs recommended).
        min_abs_weight: Threshold below which weight is considered flat (0).

    Returns:
        pd.Series: Summary including count, hit rate, mean/median P&L per trade, and avg days per trade.
    """
    trades = compute_trades(weight, returns, min_abs_weight=min_abs_weight)
    if trades.empty:
        return pd.Series(
            {
                "Trades": 0,
                "HitRate_trades": np.nan,
                "MeanPnL_per_trade": np.nan,
                "MedianPnL_per_trade": np.nan,
                "AvgDays_per_trade": np.nan,
            }
        )
    return pd.Series(
        {
            "Trades": int(trades.shape[0]),
            "HitRate_trades": float(trades["hit"].mean()),
            "MeanPnL_per_trade": float(trades["pnl"].mean()),
            "MedianPnL_per_trade": float(trades["pnl"].median()),
            "AvgDays_per_trade": float(trades["n_days"].mean()),
        }
    )


def long_short_attribution(
    returns: pd.Series,
    weight: pd.Series,
    min_abs_weight: float = 1e-6,
) -> pd.Series:
    """Attribute P&L to long vs short days by sign of weight.

    Args:
        returns: Per-period returns (after costs recommended).
        weight: Strategy weights (aligned to returns).
        min_abs_weight: Absolute weight below which position is considered flat.

    Returns:
        pd.Series: Aggregates for long/short contributions and counts.
    """
    r = clean_series(returns)
    w = pd.Series(weight).astype("float64").reindex(r.index)
    sign = np.sign(w.where(w.abs() >= min_abs_weight, 0.0))

    mask_long = sign > 0
    mask_short = sign < 0

    pnl_long = float(r[mask_long].sum())
    pnl_short = float(r[mask_short].sum())
    days_long = int(mask_long.sum())
    days_short = int(mask_short.sum())

    return pd.Series(
        {
            "PnL_long": pnl_long,
            "PnL_short": pnl_short,
            "Days_long": days_long,
            "Days_short": days_short,
            "AvgPnL_day_long": pnl_long / days_long if days_long else np.nan,
            "AvgPnL_day_short": pnl_short / days_short if days_short else np.nan,
        }
    )


# Regime stability

def yearly_summary(returns: pd.Series) -> pd.DataFrame:
    """Compute yearly CAGR, Sharpe, and Max Drawdown on after-cost returns.

    Args:
        returns: Per-period after-cost returns.

    Returns:
        pd.DataFrame: One row per year with metrics (index = Year).
    """
    r = clean_series(returns)
    if r.empty:
        return pd.DataFrame(columns=["CAGR", "Sharpe", "MaxDD"])
    
    # Ensure DatetimeINdex
    if not isinstance(r.index, pd.DatetimeIndex):
        try: 
            r.index = pd.DatetimeIndex(r.index, errors="raise")
        except:
            raise TypeError(
                "yearly_summary expects a date-indexed return series."
                f"Got index type={type(r.index).__name__}, dtype={getattr(r.index, 'dtype', None)}."
            )
    
    # Drop timezone to allow .year access consistently
    if r.index.tz is not None:
        r.index = r.index.tz_localize(None)

    rows = []
    for year, r_y in r.groupby(r.index.year):
        if r_y.empty:
            continue
        w = wealth_curve(r_y, starting_wealth=1.0)
        mdd, _, _ = max_drawdown(w)
        rows.append({
            "Year": int(year), 
            "CAGR": annualized_return(r_y), 
            "Sharpe": sharpe_ratio(r_y), 
            "MaxDD": mdd
        })
    df = pd.DataFrame(rows)
    return df.set_index("Year") if not df.empty else df


def regime_by_realized_vol(
    returns: pd.Series,
    window: int = 20,
    thresholds: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Split performance by realised-volatility regimes (computed from past returns only).

    Args:
        returns: After-cost per-period returns indexed by date.
        window: Number of days used to compute the rolling standard deviation of past
            returns; this is treated as the realised volatility at time t (uses data
            up to and including t).
        thresholds: Optional pair (low_cutoff, high_cutoff) for realised volatility.
            If None, the cut-offs are chosen automatically as the lower and upper
            third of the sample distribution of realised volatility.

    Returns:
        pd.DataFrame: One row per regime ('low', 'mid', 'high') with columns:
            - CAGR: Annualised return.
            - Sharpe: Annualised Sharpe ratio.
            - MaxDD: Maximum drawdown (negative number).
            - Count: Number of days in the regime.

    Notes:
        By default, the regime cut-offs are derived from the whole sample. If you
        prefer fixed cut-offs, pass them explicitly via `thresholds`.
    """
    ...

    r = clean_series(returns)
    if r.empty:
        return pd.DataFrame(columns=["CAGR", "Sharpe", "MaxDD", "Count"])
    realized_vol = r.shift(1).rolling(window, min_periods=window).std()  # no lookahead in the estimator
    if thresholds is None:
        q_low, q_high = realized_vol.quantile([1.0 / 3.0, 2.0 / 3.0])
    else:
        q_low, q_high = thresholds

    labels = pd.cut(
        realized_vol,
        bins=[-np.inf, q_low, q_high, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )

    rows = []
    for regime in ["low", "mid", "high"]:
        mask = labels == regime
        rr = r[mask].dropna()
        if rr.empty:
            rows.append({"Regime": regime, "CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Count": 0})
            continue
        w = wealth_curve(rr, starting_wealth=1.0)
        mdd, _, _ = max_drawdown(w)
        rows.append({"Regime": regime, "CAGR": annualized_return(rr), "Sharpe": sharpe_ratio(rr), "MaxDD": mdd, "Count": int(rr.size)})
    df = pd.DataFrame(rows)
    return df.set_index("Regime") if not df.empty else df


# One-shot summary for backtest outputs

def summarize_backtest(
    backtest_frame: pd.DataFrame,
    return_column: str = "net_return",
    turnover_column: str = "turnover",
    starting_wealth: float = 1_000.0,
    periods_per_year: Optional[int] = None,
    returns_are_log: bool = False,
    label: Optional[str] = None,
    fold: Optional[str] = None,
) -> pd.Series:
    """Summarise a backtest DataFrame into key metrics.

    Args:
        backtest_frame: Output from `single_asset_backtest` (must include returns & turnover).
        return_column: Column name for per-period returns (after costs).
        turnover_column: Column name for |Δw|.
        starting_wealth: Initial wealth for the wealth curve.
        periods_per_year: Override periods per year; inferred if None.
        returns_are_log: If True, interpret returns as log returns.
        label: Optional model label for reporting.
        fold: Optional fold name for reporting.

    Returns:
        pd.Series: Summary including CAGR, Sharpe, Sortino, MaxDD, hit rate, turnover, final wealth,
            and total costs (if available).

    Raises:
        ValueError: If required columns are missing.
    """
    if return_column not in backtest_frame.columns:
        raise ValueError(f"'{return_column}' not found in backtest_frame.")
    if turnover_column not in backtest_frame.columns:
        raise ValueError(f"'{turnover_column}' not found in backtest_frame.")

    r = clean_series(backtest_frame[return_column])
    t = clean_series(backtest_frame[turnover_column])
    if r.empty:
        # Return a labeled, but empty, series to avoid downstream KeyErrors.
        return pd.Series(
            {
                "label": label,
                "fold": fold,
                "n_periods": 0,
                "CAGR": np.nan,
                "AnnVol": np.nan,
                "Sharpe": np.nan,
                "Sortino": np.nan,
                "MaxDD": np.nan,
                "HitRate": np.nan,
                "AvgTurnover": np.nan,
                "AnnTurnover": np.nan,
                "FinalWealth": np.nan,
                "TotalCost": np.nan if "transaction_cost" in backtest_frame.columns else None,
            }
        )

    ppy = periods_per_year or infer_periods_per_year(r.index)
    w = wealth_curve(r, starting_wealth=starting_wealth, returns_are_log=returns_are_log)
    mdd, _, _ = max_drawdown(w)

    out = {
        "label": label,
        "fold": fold,
        "n_periods": int(r.size),
        "CAGR": annualized_return(r, periods_per_year=ppy, returns_are_log=returns_are_log),
        "AnnVol": annualized_volatility(r, periods_per_year=ppy),
        "Sharpe": sharpe_ratio(r, periods_per_year=ppy, risk_free_rate_annual=0.0),
        "Sortino": sortino_ratio(r, periods_per_year=ppy, mar=0.0),
        "MaxDD": mdd,  # negative number (e.g., -0.25 = -25%)
        "HitRate": hit_rate(r),
        "AvgTurnover": float(t.mean()) if not t.empty else np.nan,
        "AnnTurnover": float(t.mean()) * ppy if not t.empty else np.nan,
        "FinalWealth": float(w.iloc[-1]),
    }
    if "transaction_cost" in backtest_frame.columns:
        total_cost = float(clean_series(backtest_frame["transaction_cost"]).sum())
        out["TotalCost"] = total_cost

    return pd.Series(out)