"""Download and cache OHLCV price data for any stock or crypto ticker via yfinance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV data for any yfinance-supported ticker.

    Works for stocks ("NVDA", "AAPL") and crypto ("BTC-USD", "ETH-USD").
    Returns a DataFrame with lowercase column names and a DatetimeIndex named 'timestamp'.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'. Check the ticker and date range.")

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "timestamp"

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.dropna(inplace=True)

    print(f"[{symbol}] Downloaded {len(df)} bars ({df.index.min().date()} -> {df.index.max().date()})")
    return df


def _cache_path(symbol: str, data_dir: Path, interval: str) -> Path:
    safe_name = symbol.replace("/", "_").replace("-", "_")
    suffix = f"_{interval}" if interval != "1d" else ""
    return data_dir / f"{safe_name}_prices{suffix}.csv"


def save_prices(df: pd.DataFrame, symbol: str, output_dir: str | Path,
                interval: str = "1d") -> Path:
    """Persist a price DataFrame to CSV. Returns the saved path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, output_dir, interval)
    df.to_csv(path)
    print(f"[{symbol}] Saved to {path}")
    return path


def load_prices(symbol: str, data_dir: str | Path,
                interval: str = "1d") -> pd.DataFrame:
    """Load a previously cached price CSV. Raises FileNotFoundError if missing."""
    data_dir = Path(data_dir)
    path = _cache_path(symbol, data_dir, interval)
    if not path.exists():
        raise FileNotFoundError(f"No cached data for '{symbol}' at {path}")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    print(f"[{symbol}] Loaded {len(df)} cached bars from {path}")
    return df


def get_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: str | Path,
    interval: str = "1d",
    force_download: bool = False,
) -> pd.DataFrame:
    """Load from cache if available, otherwise download and cache.

    Cache filename encodes the interval so daily and weekly data coexist:
      NVDA_prices.csv      (daily)
      NVDA_prices_1wk.csv  (weekly)
    """
    if not force_download:
        try:
            return load_prices(symbol, data_dir, interval)
        except FileNotFoundError:
            pass

    df = download_prices(symbol, start_date, end_date, interval)
    save_prices(df, symbol, data_dir, interval)
    return df
