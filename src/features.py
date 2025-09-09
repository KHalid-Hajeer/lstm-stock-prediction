#TODO: ensure consistency across files, in terms of function outputs and names
# Core Libraries
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Iterable


# Targets
def compute_log_returns(close_prices: pd.Series) -> pd.Series:
    """Compute daily log returns.
    
    r_t = log(C_t) - log(C_{t-1})

    Args:
        close (pd.Series): Close price series; must be positive and time-sorted. 

    Returns:
        pd.Series: Log return series aligned to the input index (first value NaN).
        
    Raises:
        ValueError: If any price is non-positive.
    """
    cp = pd.Series(close_prices).astype(float)
    if (cp <= 0).any():
        raise ValueError("All prices must be positive for log returns.")
    log_return = np.log(cp).diff().rename("log_return")
    return log_return

def compute_ewma_vol(returns: pd.Series, span: int = 20) -> pd.Series:
    """Compute Exponentially Weighted Moving Average (EWMA) standard deviation using data up to time t (no lookahead)

    Args:
        returns(pd.Series): Return series; must be positive and time-sorted.
        span (int, optional): EWMA span in days. Defaults to 20.

    Returns:
        pd.Series: Rolling EWMA standard deviation series aligned to the input index (first value NaN).
    """
    r = pd.Series(returns).astype(float)
    ewma_vol = r.ewm(span=span, adjust=False, min_periods=span).std().rename(f"ewma_vol_{span}")
    return ewma_vol
    
def make_targets(close_prices: pd.Series, vol_span: int = 20) -> pd.DataFrame:
    """Create raw and volatility-normalised nextday return targets.

    y_raw = r_{t+1} 
    y_vol = r_{t+1} / sigma_t    where sigma_t is the EWMA volatility computed with data <= t.
    
    Args:
        close_price (pd.Series): Close price series; must be positive and time-sorted.
        vol_periods (int): Span for EWMA volatility. Defaults to 20.

    Returns:
        pd.DataFrame: Columns:
        - log_return: Daily log return series aligned to the input index (first value NaN).
        - ewma_vol: EWMA standard deviation series aligned to the input index (first value NaN).
        - y_raw: Raw next-day log return series aligned to the input index (first value NaN).
        - y_vol: Volatility-normalised next-day log return series aligned to the input index (first value NaN).
    """
    log_return = compute_log_returns(close_prices)
    ewma_vol = compute_ewma_vol(log_return, span=vol_span)
    y_raw = log_return.shift(-1).rename("y_raw")
    y_vol = (y_raw / ewma_vol).rename("y_vol")
    return pd.concat([log_return, ewma_vol, y_raw, y_vol], axis=1)

# Price-based features
def make_price_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Compute price detas, ratios, simple returns, and momentum.

    Args:
        ohlc (pd.DataFrame): DataFrame with atleast columns: ['open', 'high', 'low', 'close'].

    Returns:
        pd.DataFrame: Price-based features.
    """
    features = pd.DataFrame(index=ohlc.index)
    # Disparities
    features["close_minus_open"] = ohlc["close"] - ohlc["open"]
    features["high_minus_low"] = ohlc["high"] - ohlc["low"]
    # Ratios
    features["close_over_open"] = ohlc["close"] / ohlc["open"]
    features["high_over_low"] = ohlc["high"] / ohlc["low"]
    
    # Returns
    features["simple_return"] = ohlc["close"].pct_change()
    features["log_return"] = compute_log_returns(ohlc["close"])
    
    # Momentum
    features["momentum_5d"] = ohlc["close"] - ohlc["close"].shift(5)
    return features

# Technical indicators
def compute_simple_moving_average(series: pd.Series, lookback: int) -> pd.Series:
    """Compute simple moving average (SMA) over a specified lookback period.

    Args:
        series (pd.Series): Time series data to compute the SMA on.
        lookback (int): Lookback period for the SMA.

    Returns:
        pd.Series: SMA series aligned to the input index (first values NaN).
    """
    sma = series.rolling(lookback, min_periods=lookback).mean().rename(f"sma_{lookback}")
    return sma

def compute_exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    """Compute exponential moving average (EMA) over a specified span.

    Args:
        series (pd.Series): Time series data to compute the EMA on.
        span (int): Span for the EMA.

    Returns:
        pd.Series: Exponential moving average series aligned to the input index (first values NaN).
    """
    ewm = series.ewm(span=span, adjust=False, min_periods=span).mean().rename(f"ema_{span}")
    return ewm

def compute_relative_strength_index(close_prices: pd.Series, lookback: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) over a specified lookback period, with EWMA smoothing.

    Args:
        close_prices (pd.Series): Close price series; must be positive and time-sorted.
        lookback (int): Lookback period for RSI calculation. Defaults to 14.

    Returns:
        pd.Series: Relative Strength Index series aligned to the input index (first values NaN), values between 0 and 100.
    """
    close_prices = pd.Series(close_prices, dtype="float64").rename("close_prices")
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    alpha = 1.0 / lookback
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=lookback).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=lookback).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)  # Avoid division by zero
    rsi = 100.0 - (100.0 / (1 + rs))
    rsi = rsi.rename(f"rsi_{lookback}")
    return rsi

def compute_macd_indicator(
        close_prices: pd.Series, 
        short_span: int = 12, 
        long_span: int = 26, 
        signal_span: int = 9
        ) -> pd.DataFrame:
    """Compute MACD (Moving Average Convergence Divergence) indicator.

    Args:
        close_prices (pd.Series): Close price series; must be positive and time-sorted.
        short_span (int): Short EMA span for MACD. Defaults to 12.
        long_span (int): Long EMA span for MACD. Defaults to 26.
        signal_span (int): Signal line EMA span for MACD. Defaults to 9.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - macd: MACD line (short EMA - long EMA).
            - signal: Signal line (EMA of MACD).
            - histogram: MACD histogram (MACD - Signal).
    """
    short_ema = compute_exponential_moving_average(close_prices, short_span)
    long_ema = compute_exponential_moving_average(close_prices, long_span)
    macd = short_ema - long_ema
    signal = compute_exponential_moving_average(macd, signal_span)
    histogram = macd - signal
    return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram})

def compute_bollinger_bands(close_prices: pd.Series, lookback: int = 20, width: float = 2.0) -> Tuple[pd.Series, pd.Series,]:
    """Compute Bollinger Bands.

    Args:
        close_prices (pd.Series): Close price series; must be positive and time-sorted.
        lookback (int): Lookback window for SMA/std. Defaults to 20.
        width (float): Width multiplier for the bands. Defaults to 2.0.
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (bb_upper_band, bb_lower_band).
    """
    close_prices = pd.Series(close_prices, dtype="float64").rename("close_prices")
    middle_band = compute_simple_moving_average(close_prices, lookback).rename(f"bb_mid_{lookback}")
    std = close_prices.rolling(lookback, min_periods=lookback).std()
    upper_band = (middle_band + width * std).rename(f"bb_upper_{lookback}")
    lower_band = (middle_band - width * std).rename(f"bb_lower_{lookback}")
    return middle_band, upper_band, lower_band

# Combine all technical features
def make_technical_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Compute a suite of technical indicators.

    Args:
        ohlc (pd.DataFrame): DataFrame with atleast columns: ['open', 'high', 'low', 'close'].

    Returns:
        pd.DataFrame: Technical indicator features (no central lag applied yet).
    """
    features = pd.DataFrame(index=ohlc.index)
    
    # Moving Averages
    features["sma_10"] = compute_simple_moving_average(ohlc["close"], 10)
    features["sma_20"] = compute_simple_moving_average(ohlc["close"], 20)
    features["ema_20"] = compute_exponential_moving_average(ohlc["close"], 20)
    features["ema_50"] = compute_exponential_moving_average(ohlc["close"], 50)
    
    # RSI
    features["rsi_14"] = compute_relative_strength_index(ohlc["close"], 14)
    
    # MACD
    macd_df = compute_macd_indicator(ohlc["close"], 12, 26, 9)
    features["macd"] = macd_df["macd"]
    features["macd_signal"] = macd_df["signal"]
    features["macd_histogram"] = macd_df["histogram"]
    
    # Bollinger Bands
    bb_middle, bb_upper, bb_lower = compute_bollinger_bands(ohlc["close"], 20, 2.0)
    features["bb_middle_20"] = bb_middle
    features["bb_upper_20"] = bb_upper
    features["bb_lower_20"] = bb_lower

    # Stochastic Oscillator %K
    lowest_low_14 = ohlc["low"].rolling(14, min_periods=14).min()
    highest_high_14 = ohlc["high"].rolling(14, min_periods=14).max()
    features["stochastic_k_14"] = ((ohlc["close"] - lowest_low_14) / (highest_high_14 - lowest_low_14) * 100.0)
    return features

# Volatility & Rolling Statistics
def make_volatility_features(ohlc: pd.Series, windows: list[int] = [10, 20]) -> pd.DataFrame:
    """Compute volatility from returns, rolling means/stds, and a z-score.

    Args:
        ohlc (pd.Series): Return series; must be time-sorted.
        windows (list[int], optional): List of window sizes for rolling calculations. Defaults to [10, 20].

    Returns:
        pd.DataFrame: Volatility and rolling mean features (no central lag applied yet).
    """
    features = pd.DataFrame(index=ohlc.index)
    log_return = compute_log_returns(ohlc["close"])
    for window in windows:
        features[f"vol_{window}"] = log_return.rolling(window, min_periods=window).std()
        features[f"rolling_std_{window}"] = ohlc["close"].rolling(window, min_periods=window).std()
        features[f"rolling_mean_{window}"] = ohlc["close"].rolling(window, min_periods=window).mean()
    features["z_score_10"] = (ohlc["close"] - features["rolling_mean_10"]) / features["rolling_std_10"]
    return features

# Volume-based features
def on_balance_volume(
        close_prices: pd.Series, 
        volumes: pd.Series, 
        check_nonnegative: bool = True, 
        align: str = "inner"
        ) -> pd.Series:
    """Compute On-Balance Volume (OBV) in a vectorized, leak-safe way.

    OBV_t = OBV_{t-1} + Vol_t * sign(Close_t - Close_{t-1}),
    with OBV_0 = 0

    Args:
        close_prices (pd.Series): Close price series; time-sorted.
        volumes (pd.Series): Volume series; time-sorted.
        check_nonnegative(bool): if True, raises ValueError if any volume is negative.
        align (str): How to align indices if they differ. Options: 'inner', 'outer', 'left', 'right'. Defaults to 'inner'.

    Returns:
        pd.Series: OBV series aligned to the chosen alignment, named "obv".
        
    Raises:
        ValueError: If negative volumes found (when check_nonnegative is True).
    """
    # Align by index (handles missing timestamps) 
    close_prices, volumes = close_prices.align(volumes, join=align)

    # Coerce to float and basic checks
    close_prices = pd.Series(close_prices, dtype="float64")
    volumes = pd.Series(volumes, dtype="float64")
    if check_nonnegative and (volumes < 0).any():
        raise ValueError("All volume values must be non-negative.")
    
    # Vectorized OBV: sign of price change times today's volume, cumulative sum
    price_change_sign = np.sign(close_prices.diff().fillna(0.0))
    obv = (price_change_sign * volumes).cumsum().rename("obv")
    return obv

def make_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based features, and VWAP and trade_count features when available.

    Args:
        data (pd.DataFrame): DataFrame with atleast columns: ['close', 'volume'], optionally ['vwap', 'trade_count'].

    Returns:
        pd.DataFrame: Volume-based features (no central lag applied yet).
    """
    features = pd.DataFrame(index=data.index)
    if "volume" in data.columns:
        features["volume_change"] = data["volume"].pct_change()
        features["volume_over_avg_10d"] = data["volume"] / data["volume"].rolling(10, min_periods=10).mean()
        features["obv"] = on_balance_volume(data["close"], data["volume"])
    if "vwap" in data.columns:
        features["close_minus_vwap"] = data["close"] - data["vwap"]
        features["close_over_vwap"] = data["close"] / data["vwap"]
    if "trade_count" in data.columns:
        features["trade_count_change"] = data["trade_count"].pct_change()
        features["trade_count_over_avg_10d"] = data["trade_count"] / data["trade_count"].rolling(10, min_periods=10).mean()
    return features

# Statistifcal features

def compute_shannon_entropy_window(
        window_values: pd.Series, 
        bins: int = 20, 
        base: Optional[float] = None, 
        min_count: int = 5
        ) -> float:
    """Compute Shannon entropy of a window using histogram counts.

    H(X) = - sum(p(x) * log_b(p(x))) for all unique x in X

    Args:
        window_values (pd.Series): Window of values (NaNs allowed; they are dropped).
        bins(int): Number of equal=width bins to use for histogram.
        base (Optional[float]): Log base. If None, use natural log. Defaults to None.
        min_count (int): Minimum number of non-NaN values required to compute entropy.

    Returns:
        float: Shannon entropy of the window, np.nan if not enough data.
    """
    window_values = pd.Series(window_values).dropna().to_numpy()
    if window_values.size < min_count:
        return float("nan")
    counts, _ = np.histogram(window_values, bins=bins, density=False)
    total = counts.sum()
    if total == 0:
        return float("nan")
    probabilities = counts[counts > 0].astype(float) / float(total)
    entropy = -(probabilities * np.log(probabilities)).sum()
    if base is not None:
        entropy /= np.log(base)
    return float(entropy)

def compute_hurst_exponent_window(
        window_values: pd.Series, 
        min_len: int = 20, 
        max_lag: int = 19
        ) -> float:
    """Compute Hurst exponent approximation, via log-log slope of lagged-diff std vs lag.

    Args:
        window_values (pd.Series): Window of values (NaNs allowed; they are dropped).
        min_len (int): Minimum required non-NaN observations.
        max_lag (int): Maximum lag to use for Hurst exponent computation.

    Returns:
        float: Hurst exponent of the window in [~0, ~1]; np.nan if not enough data.
    """
    window_values = pd.Series(window_values).dropna().to_numpy()
    if window_values.size < min_len:
        return float("nan")
    lags = np.arange(2, max_lag + 1, dtype=int)
    lags = lags[lags < window_values.size]
    if lags.size == 0:
        return float("nan")
    # std of lagged differences
    tau = np.array([np.std(window_values[lag:] - window_values[:-lag], ddof=0) for lag in lags])
    # guard against non-positive/NaN values
    mask = np.isfinite(tau) & (tau > 0)
    if mask.sum() < 3:
        return float("nan")
    slope = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]
    return slope

def compute_rolling_autocorr(series: pd.Series, lag: int, window: int) -> pd.Series:
    """Compute rolling autocorrelation at a specified lag using a normalised dot product.

    Args:
        series (pd.Series): Input series (time-sorted).
        lag (int): Autocorrelation lag (>= 1).
        window (int): Rolling window size.

    Returns:
        pd.Series: Rolling autocorrelation series at 'lag' with the same index as 'series'.
    
    Raises:
        ValueError: If lag < 1.
    """
    if lag <= 0:
        raise ValueError("lag must be >=1")
    
    def _autocorr_block(x: np.ndarray) -> float:
        """Helper to compute autocorr for a single block."""
        # X is a numpy array (raw=True); length == window\
        if x.size <= lag or np.isnan(x).any():
            return np.nan
        a = x[:-lag] - x[:-lag].mean()
        b = x[lag:] - x[lag:].mean()
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        return float(np.sum(a * b) / denom) if denom > 0 else np.nan
    
    autocorr = series.rolling(window, min_periods=window).apply(_autocorr_block, raw=True).rename(f"autocorr_lag_{lag}_w{window}")
    return autocorr

def make_statistical_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Compute statistical features.

    Args:
        ohlc (pd.DataFrame): DataFrame with atleast column ["close"].

    Returns:
        pd.DataFrame: Statistical features.
    """
    features = pd.DataFrame(index=ohlc.index)
    features["entropy_close_20"] = ohlc["close"].rolling(20, min_periods=20).apply(lambda w: compute_shannon_entropy_window(w,bins=20, base=None, min_count=5), raw=False)
    features["hurst_close_100"] = ohlc["close"].rolling(100, min_periods=100).apply(lambda w: compute_hurst_exponent_window(w, min_len=20, max_lag=19), raw=False)
    
    for lag in (1, 2, 3):
        features[f"autocorr_close_{lag}_w20"] = compute_rolling_autocorr(ohlc["close"], lag, window=20)
    return features

# Orchestrator (central lag)
def build_feature_matrix(ohlcv: pd.DataFrame, lag_days: int = 1, drop_na_rows: bool = True) -> pd.DataFrame:
    """ Build a lag-safe feature matrix from OHLCV (and optional VWAP and trade_count).

    Computes category featuresm concatenates them, then applies a single central lag
    ('lag_days', default 1) so features at time t are used to predict y at time t_1.
    This prevents look-ahead leakage.

    Args:
        ohlcv(pd.DataFrame): Input OHCLV frame; index should be a DateTimeIndex.
            Must include ['open', 'high', 'low', 'close', 'volume]; optional ['vwap', 'trade_count'].
        lag_days (int): Central lag to apply to tehe entire feature matrix. Defaults to 1.
        drop_na_rows (bool):If True, drop rows with NaNs after lagging.

    Returns:
        pd.DataFrame: Feature matrix aligned to the input index with a central lag applied.
    """
    feature_blocks = [
        make_price_features(ohlcv),
        make_technical_features(ohlcv),
        make_volume_features(ohlcv),
        make_volatility_features(ohlcv),
        make_statistical_features(ohlcv)
    ]
    features = pd.concat(feature_blocks, axis=1)

    # Replace inf with NaN before shifting
    features = features.replace([np.inf, -np.inf], np.nan)

    # Central alg to enforce t -> t+lag_days prediction mapping
    if lag_days and lag_days > 0:
        features = features.shift(lag_days)
    if drop_na_rows:
        features = features.dropna()
    
    return features