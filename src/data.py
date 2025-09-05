# Core Libraries
from __future__ import annotations
import os
import datetime as dt
from typing import Optional, Union

import pandas as pd
from dateutil import relativedelta

# Alpaca Libraries
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Internal helpers
def _client(api_key: Optional[str] = None, secret_key: Optional[str] = None) -> StockHistoricalDataClient:
    """Create an Alpaca historical data client.

    Args:
        api_key (Optional[str]): Alpaca API key. If None, reads ALPACA_API_KEY from the .env.
        secret_key (Optional[str]): Alpaca secret key. If None, reads ALPACA_SECRET_KEY from the .env.

    Returns:
        StockHistoricalDataClient: Initialised Alpaca historical data client.

    Raises:
        ValueError: If api_key or secret_key is not provided and environment variables are not set.
    """
    api = api_key or os.getenv('ALPACA_API_KEY')
    sec = secret_key or os.getenv('ALPACA_SECRET_KEY')
    if not api or not sec:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment variables.")
    return StockHistoricalDataClient(api, sec)

def _resolve_feed(feed: Optional[Union[str, object]]) -> Optional[object]:
    """Return a feed compatible with alpaca-py (IEX by default for free plan)
    
    Args:
        feed (Optional[Union[str, object]]): Feed name or feed object.

    Returns:
        Optional[object]: Feed object.
    """
    if feed is None:
        feed = "iex"
    if isinstance(feed, str):
        feed_str = feed.strip().lower()
        # If enumerations unavailable, pass the raw string
        return "iex" if feed_str in ("iex", "free") else "sip"
    # If caller passed an enumeration already:
    return feed

def _flatten_bars_df(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Alpaca bars into a clean OHLCV(+optional VWAP & trade_count) DataFrame.
    
    Alpaca may return a MultiIndex (symbol, timestamp). This flattents it, sets a UTC DatetimeIndex, 
    standardises column names, enforces numeric types, removes duplicates, and drops rows with missing values.

    Args:
        bars_df (pd.DataFrame): Raw bars DataFrame from the Alpaca SDK.

    Returns:
        pd.DataFrame: Clean frame indexed by UTC timestamps with available columns among:
        ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count'].

    Raises:
        ValueError: If the DataFrame does not contain the expected columns.
    """
    # Always bring the time index out as a column and call it 'timestamp'
    if isinstance(bars_df.index, pd.MultiIndex):
        df = bars_df.reset_index()
        # If Alpaca already names the time level 'timestamp', great; otherwise use the last index level name
        time_col = 'timestamp' if 'timestamp' in df.columns else bars_df.index.names[-1]
    else:
        # Handles DatetimeIndex and others uniformly
        df = bars_df.reset_index()
        # If the index had no name, reset_index creates a column called 'index'
        time_col = bars_df.index.name or 'index'

    df = df.rename(columns={time_col: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.set_index('timestamp').sort_index()

    
    # Standardise column names
    df.columns = [col.lower() for col in df.columns]
    
    # Ensure numeric types
    keep = [c for c in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count'] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors='coerce')
    df = df[~df.index.duplicated(keep='first')].dropna()
    
    # Check if required columns are present
    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}. Found: {df.columns.tolist()}")
    
    return df
    
def get_stock_data(
    symbol: str = 'SPY',
    years: int = 10,
    path: str = '/data/raw/SPY.parquet',
    force_refresh: bool = False,
    feed: str = 'iex' # Default to IEX (free tier)
    ) -> pd.DataFrame:
    """ Fetch daily OHLCV (+ vwap & trade_count) data for a given stock symbol from Alpaca.
    
    If a cache file exists at 'path' and 'force_refresh' is False, the function loads from cache.
    Otherwise, it fetches the data from Alpaca's API, cleans the result, and saves to the cache.
    
    Args:
        symbol (str): Ticker symbol to fetch (e.g., 'SPY').
        years (int): Number of past calendar years to retrieve data for (default is 10).
        path (str): Cache file path (.parquet). Used for both saving and loading.
        force_refresh (bool): If True, ignore any cache and fetch fresh data.

    Returns:
        pd.DataFrame: DataFrame indexed by UTC timestamps, with columns:
            ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count'].

    Raises:
        ValueError: If 'path' has an unsupported extension on load.
        RuntimeError: If ALPACA_API_KEY or ALPACA_SECRET_KEY are missing.
    """    
    # Load from cache if available and force_refresh is False
    if (not force_refresh) and os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.parquet':
            return pd.read_parquet(path)
        elif ext == '.csv':
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # Initialize the Alpaca client
    client = _client()
    end = dt.datetime.now(dt.timezone.utc)  # Current time in UTC
    start = end - relativedelta.relativedelta(years=years)  # Calculate start time for the data request

    # Define the parameters for the data request
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=_resolve_feed(feed),
        limit=10000
    )

    # Fetch the stock data from Alpaca with auto fallback
    try:
        bars = client.get_stock_bars(request_params)
    except Exception as e:
        if "subscription" in str(e).lower() and "sip" in str(e).lower():
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=_resolve_feed("iex"),
                limit=10000,
            )
            bars = client.get_stock_bars(req)
        else:
            raise


    # Clean the DataFrame
    df = _flatten_bars_df(bars.df)
    
    # Save the DataFrame to a parquet file at path 
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.lower().endswith(".parquet"):
            df.to_parquet(path)
    return df

def align_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """ Light cleanup + alignment of time index.
    
    Ensures a DatetimeIndex (UTC), is sorted and unique, and returns only standard columns.
    No forward-filling is performed.
        
    Args:
        df (pd.DataFrame): Raw or cached DataFrame with a 'timestamp' column or index.
        
    Returns:
        pd.DataFrame: Clean OHLCV (+ VWAP & trade_count) DataFrame with a unique, sorted UTC index.
    
    Raises:
        ValueError: If 'timestamp' column or index is missing or not in datetime format.
    """
    df = df.copy()

    # Establish the time index
    if 'timestamp' in df.columns:
        idx = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    else:
        # Try common alternatives
        for cand in ('time', 'date', 'datetime'):
            if cand in df.columns:
                idx = pd.to_datetime(df[cand], utc=True, errors='coerce')
                break
        else:
            raise KeyError("No 'timestamp' column or DatetimeIndex found.")

    df.index = idx
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep='first')].sort_index()

    keep = [c for c in ('open','high','low','close','volume','vwap','trade_count') if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors='coerce').dropna(how='any')

    return df