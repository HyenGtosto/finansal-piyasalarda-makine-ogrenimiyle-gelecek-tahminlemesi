"""Load locally-stored raw price CSVs (from the friend's pipeline) as OHLCV DataFrames.

The friend's price files contain hourly tick data with columns:
    datetime_utc, price_usd, market_cap_usd, total_volume_usd

This module aggregates them to daily or 4-hour OHLCV bars so the sentiment
dates align perfectly with the price data (both span Jun 2025 - Jun 2026).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Maps a yfinance-style symbol to a local raw CSV path (relative to project root)
LOCAL_PRICE_FILES: dict[str, str] = {
    "BTC-USD": "data/raw/bitcoin_price_raw.csv",
    # Add more as the friend's pipeline downloads them:
    # "ETH-USD": "data/raw/ethereum_price_raw.csv",
}


def load_local_prices(
    csv_path: str | Path,
    interval: str = "1d",
    datetime_col: str = "datetime_utc",
    price_col: str = "price_usd",
    volume_col: str = "total_volume_usd",
) -> pd.DataFrame:
    """Aggregate a raw hourly price CSV to OHLCV bars at the requested interval.

    Args:
        interval: "1d" for daily bars, "4h" for 4-hour bars.

    Returns a DataFrame with columns [open, high, low, close, volume]
    and a timezone-naive DatetimeIndex named 'timestamp'.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in [datetime_col, price_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df = df.sort_values(datetime_col)

    if interval == "4h":
        df["_bucket"] = df[datetime_col].dt.floor("4h").dt.tz_localize(None)
        label = "4H"
    else:
        df["_bucket"] = df[datetime_col].dt.normalize().dt.tz_localize(None)
        label = "daily"

    vol_agg = (volume_col, "sum") if volume_col in df.columns else (price_col, "count")
    bars = df.groupby("_bucket").agg(
        open=(price_col, "first"),
        high=(price_col, "max"),
        low=(price_col, "min"),
        close=(price_col, "last"),
        volume=vol_agg,
    )
    bars.index.name = "timestamp"
    bars = bars.dropna()

    print(
        f"[local] Aggregated {len(df)} hourly rows -> {len(bars)} {label} bars "
        f"({bars.index.min().date()} -> {bars.index.max().date()})"
    )
    return bars


def has_local_prices(symbol: str, project_root: Path) -> bool:
    """Return True if a local raw CSV exists for this symbol."""
    rel_path = LOCAL_PRICE_FILES.get(symbol)
    if rel_path is None:
        return False
    return (project_root / rel_path).exists()


def get_local_prices(symbol: str, project_root: Path, interval: str = "1d") -> pd.DataFrame:
    """Load and aggregate the local raw price CSV for a symbol.

    Raises KeyError if the symbol has no registered local file.
    Raises FileNotFoundError if the file doesn't exist on disk.
    """
    rel_path = LOCAL_PRICE_FILES.get(symbol)
    if rel_path is None:
        raise KeyError(f"No local price file registered for '{symbol}'. "
                       f"Add it to LOCAL_PRICE_FILES in load_local_prices.py")

    csv_path = project_root / rel_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Local price file not found: {csv_path}")

    return load_local_prices(csv_path, interval=interval)
