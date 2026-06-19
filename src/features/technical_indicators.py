"""Compute technical indicators and build the final feature DataFrame for any OHLCV asset."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual indicator functions — each takes a DataFrame, returns it back
# ---------------------------------------------------------------------------

def add_sma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Simple Moving Averages expressed as ratio to current close.

    SMA{w}_ratio = SMA{w} / close.  Values cluster near 1.0 and are
    stationary across price regimes, so MinMaxScaler generalises to
    out-of-sample price levels (e.g. NVDA $30 in 2018 vs $1400 in 2025).
    """
    if windows is None:
        windows = [5, 20, 50]
    df = df.copy()
    for w in windows:
        df[f"SMA{w}_ratio"] = df["close"].rolling(window=w).mean() / df["close"]
    return df


def add_ema(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Exponential Moving Averages expressed as ratio to current close."""
    if windows is None:
        windows = [12, 20, 26]
    df = df.copy()
    for w in windows:
        df[f"EMA{w}_ratio"] = df["close"].ewm(span=w, adjust=False).mean() / df["close"]
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index (RSI)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"RSI{window}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD, Signal, and Histogram expressed as % of current close price.

    Dividing by close makes them stationary across different price regimes.
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD_pct"] = (ema_fast - ema_slow) / df["close"]
    df["MACD_Signal_pct"] = df["MACD_pct"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist_pct"] = df["MACD_pct"] - df["MACD_Signal_pct"]
    return df


def add_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands: upper, middle, lower, and percentage width."""
    df = df.copy()
    sma = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["BB_Upper"] = sma + n_std * std
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - n_std * std
    # Width relative to middle band — a volatility proxy
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived price features: pct change, log return, high-low range."""
    df = df.copy()
    df["Price_Change"] = df["close"].pct_change()
    df["Log_Return"] = np.log(df["close"] / df["close"].shift(1))
    df["HL_Pct"] = (df["high"] - df["low"]) / df["close"]
    return df


def add_derived_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Crossover and overbought/oversold signals derived from existing indicators.

    MACD_Cross  :  1 when MACD line crossed above Signal this bar (golden cross)
                  -1 when MACD line crossed below Signal (death cross), 0 otherwise
    RSI_Zone    :  1 = overbought (RSI > 70), -1 = oversold (RSI < 30), 0 = neutral
    Price_Above_SMA20: 1 when close is above SMA20, -1 when below
    """
    df = df.copy()

    if "MACD_pct" in df.columns and "MACD_Signal_pct" in df.columns:
        above_now  = df["MACD_pct"] > df["MACD_Signal_pct"]
        above_prev = above_now.shift(1).fillna(False).astype(bool)
        df["MACD_Cross"] = 0
        df.loc[ above_now & ~above_prev, "MACD_Cross"] =  1   # just crossed up
        df.loc[~above_now &  above_prev, "MACD_Cross"] = -1   # just crossed down

    if "RSI14" in df.columns:
        df["RSI_Zone"] = 0
        df.loc[df["RSI14"] > 70, "RSI_Zone"] =  1
        df.loc[df["RSI14"] < 30, "RSI_Zone"] = -1

    if "SMA20_ratio" in df.columns:
        # SMA20_ratio = SMA20/close; ratio < 1 means close > SMA20 (price above MA)
        df["Price_Above_SMA20"] = (df["SMA20_ratio"] < 1).astype(int) * 2 - 1

    return df


def add_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add classification and regression target columns.

    Target_Class : 1 if close goes up in `horizon` bars, else 0
    Target_Return: forward return over `horizon` bars
    Next_Close   : raw next close price (for regression)
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["Target_Class"] = (future_close > df["close"]).astype(int)
    df["Target_Return"] = (future_close - df["close"]) / df["close"]
    df["Next_Close"] = future_close
    return df


def add_sentiment(
    df: pd.DataFrame,
    sentiment_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Merge a daily sentiment score into the feature DataFrame.

    If sentiment_series is None or empty, fills the column with 0.0
    so the rest of the pipeline works unchanged until VADER is wired in.
    sentiment_series should be a pd.Series with a DatetimeIndex.
    """
    df = df.copy()
    if sentiment_series is not None and not sentiment_series.empty:
        aligned = sentiment_series.reindex(df.index, method="ffill").fillna(0.0)
        df["Sentiment"] = aligned.values
    else:
        df["Sentiment"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    sentiment_series: pd.Series | None = None,
    target_horizon: int = 1,
    sma_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline on any OHLCV DataFrame.

    Applies all indicators, merges sentiment (zeros if not provided),
    adds targets, then drops NaN rows created by rolling windows.
    """
    df = add_sma(df, sma_windows)
    df = add_ema(df, ema_windows)
    df = add_rsi(df, rsi_window)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_price_features(df)
    df = add_derived_signals(df)
    df = add_sentiment(df, sentiment_series)
    df = add_targets(df, target_horizon)
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Feature set definitions (used by ablation scenarios)
# ---------------------------------------------------------------------------

FEATURES_PRICE_ONLY = [
    "close",
    "volume",
]

FEATURES_TECHNICAL = [
    "close",
    "volume",
    "SMA5_ratio",
    "SMA20_ratio",
    "SMA50_ratio",
    "EMA12_ratio",
    "EMA20_ratio",
    "EMA26_ratio",
    "RSI14",
    "MACD_pct",
    "MACD_Signal_pct",
    "MACD_Hist_pct",
    "BB_Width",
    "Price_Change",
    "HL_Pct",
    "MACD_Cross",
    "RSI_Zone",
    "Price_Above_SMA20",
]

FEATURES_WITH_SENTIMENT = FEATURES_TECHNICAL + ["Sentiment"]
