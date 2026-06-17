"""VADER sentiment scoring and daily aggregation for tweet datasets.

Designed around the Bitcoin tweet data in data/interim/, but works on any
DataFrame that has a text column and a datetime column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Tweet-level scoring
# ---------------------------------------------------------------------------

def score_tweet(text: str) -> tuple[float, str]:
    """Return (compound_score, label) for a single text string.

    compound is in [-1, +1].
    label is 'positive', 'negative', or 'neutral'.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, "neutral"

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return compound, label


def score_tweets(
    df: pd.DataFrame,
    text_col: str = "text",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run VADER on every row and add sentiment_score / sentiment_label columns.

    Works on any DataFrame with a text column — does NOT require tweet-specific columns.
    Returns a copy; original DataFrame is not modified.
    """
    df = df.copy()
    n = len(df)

    compounds, labels = [], []
    for i, text in enumerate(df[text_col]):
        c, l = score_tweet(text)
        compounds.append(c)
        labels.append(l)
        if verbose and (i + 1) % 5000 == 0:
            print(f"  Scored {i + 1:,}/{n:,} tweets...")

    df["sentiment_score"] = compounds
    df["sentiment_label"] = labels
    return df


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = "created_at",
    score_col: str = "sentiment_score",
    weight_col: str | None = "engagement_weight",
) -> pd.DataFrame:
    """Aggregate tweet-level scores to one row per calendar day.

    Returns a DataFrame indexed by date with columns:
        sentiment_mean, sentiment_std,
        positive_ratio, negative_ratio, neutral_ratio,
        tweet_count,
        weighted_sentiment_mean  (only if weight_col is present and not None)

    The index is a DatetimeIndex at day granularity (timezone-naive).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["_date"] = df[date_col].dt.normalize().dt.tz_localize(None)

    has_weights = (
        weight_col is not None
        and weight_col in df.columns
        and df[weight_col].notna().any()
    )

    grouped = df.groupby("_date")

    agg: dict[str, pd.Series] = {
        "sentiment_mean": grouped[score_col].mean(),
        "sentiment_std":  grouped[score_col].std().fillna(0.0),
        "tweet_count":    grouped[score_col].count(),
    }

    # Direction ratios
    if "sentiment_label" in df.columns:
        for sentiment_class in ("positive", "negative", "neutral"):
            agg[f"{sentiment_class}_ratio"] = (
                grouped["sentiment_label"]
                .apply(lambda g: (g == sentiment_class).mean())
            )

    daily = pd.DataFrame(agg)

    # Weighted sentiment
    if has_weights:
        def _weighted_mean(grp: pd.DataFrame) -> float:
            w = grp[weight_col]
            s = grp[score_col]
            total_w = w.sum()
            return (s * w).sum() / total_w if total_w > 0 else 0.0

        daily["weighted_sentiment_mean"] = (
            df.groupby("_date").apply(_weighted_mean)
        )

    daily.index.name = "date"
    return daily


def daily_sentiment_series(
    df: pd.DataFrame,
    date_col: str = "created_at",
    score_col: str = "sentiment_score",
    use_weighted: bool = True,
    weight_col: str | None = "engagement_weight",
) -> pd.Series:
    """Return a single pd.Series of daily sentiment scores, indexed by date.

    Suitable for passing directly into build_features(sentiment_series=...).
    Uses weighted_sentiment_mean if available and use_weighted=True, else sentiment_mean.
    """
    daily = aggregate_daily_sentiment(df, date_col, score_col, weight_col)

    if use_weighted and "weighted_sentiment_mean" in daily.columns:
        series = daily["weighted_sentiment_mean"]
    else:
        series = daily["sentiment_mean"]

    series.name = "Sentiment"
    return series


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_and_score_tweets(
    cleaned_csv: str | Path,
    sentiment_csv: str | Path,
    text_col: str = "text",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the cleaned tweet CSV, score with VADER, merge engagement weights
    from the existing sentiment CSV, and return the combined DataFrame.

    The existing sentiment CSV (with pre-computed engagement_weight) is used
    as the base; only sentiment_score and sentiment_label are overwritten.
    """
    cleaned = pd.read_csv(cleaned_csv)
    existing = pd.read_csv(sentiment_csv)

    if verbose:
        print(f"Scoring {len(cleaned):,} tweets with VADER...")

    scored = score_tweets(cleaned[[col for col in cleaned.columns if col != "sentiment_score"]], text_col, verbose)

    # Merge engagement weights from existing file (already computed by friend's pipeline)
    if "engagement_weight" in existing.columns:
        weights = existing[["tweet_id", "engagement_weight"]]
        scored = scored.merge(weights, on="tweet_id", how="left")

    return scored


def save_daily_sentiment(
    daily: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save the daily sentiment aggregation DataFrame to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily = daily.copy()
    if "tweet_count" in daily.columns:
        daily["tweet_count"] = daily["tweet_count"].astype(int)
    daily.to_csv(out, float_format="%.4f")
    print(f"Daily sentiment saved -> {out}  ({len(daily)} days)")
    return out


def load_daily_sentiment_series(
    csv_path: str | Path,
    use_weighted: bool = True,
) -> pd.Series:
    """Load a previously saved daily sentiment CSV and return as a pd.Series.

    Tries weighted_sentiment_mean first (if use_weighted=True), else falls
    back to sentiment_mean.
    """
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)

    if use_weighted and "weighted_sentiment_mean" in df.columns:
        series = df["weighted_sentiment_mean"]
    elif "sentiment_mean" in df.columns:
        series = df["sentiment_mean"]
    else:
        raise ValueError(f"No usable sentiment column found in {csv_path}. Columns: {df.columns.tolist()}")

    series.name = "Sentiment"
    return series


# ---------------------------------------------------------------------------
# NVDA weekly sentiment (from Kaggle finvader daily scores)
# ---------------------------------------------------------------------------

def process_nvda_weekly_sentiment(
    raw_csv: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Aggregate daily finvader scores to weekly and save to processed CSV.

    Reads data/raw/nvidia_stock_sentiment.csv (date, finvader, high, low, adjusted_close).
    Groups by ISO week (Monday anchor) and takes the mean finvader for the week.
    Saves to data/processed/nvidia_weekly_sentiment.csv.
    """
    df = pd.read_csv(raw_csv, parse_dates=["date"])
    df = df.sort_values("date")

    # Floor each trading day to the Monday of its week
    df["week"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    df["week"] = df["week"].dt.normalize()

    weekly = df.groupby("week").agg(
        sentiment_mean=("finvader", "mean"),
        trading_days=("finvader", "count"),
    ).reset_index().rename(columns={"week": "date"})
    weekly = weekly.set_index("date")
    weekly["trading_days"] = weekly["trading_days"].astype(int)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out, float_format="%.4f")
    print(f"NVDA weekly sentiment saved -> {out}  ({len(weekly)} weeks)")
    print(f"Date range: {weekly.index.min().date()} -> {weekly.index.max().date()}")
    return weekly


def load_nvda_weekly_sentiment_series(
    csv_path: str | Path,
) -> pd.Series:
    """Load processed nvidia_weekly_sentiment.csv and return as a pd.Series."""
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    series = df["sentiment_mean"]
    series.name = "Sentiment"
    return series


# ---------------------------------------------------------------------------
# 4-hour sentiment aggregation
# ---------------------------------------------------------------------------

def aggregate_4h_sentiment(
    df: pd.DataFrame,
    date_col: str = "created_at",
    score_col: str = "sentiment_score",
    weight_col: str | None = "engagement_weight",
) -> pd.DataFrame:
    """Group tweets into 4-hour UTC buckets and compute weighted sentiment.

    Args:
        df:         DataFrame with per-tweet sentiment scores and timestamps.
        date_col:   Column with tweet UTC timestamp (ISO string or datetime).
        score_col:  VADER compound score column.
        weight_col: Optional engagement weight column for weighted mean.

    Returns a DataFrame indexed by 4H bucket (timezone-naive) with columns:
        sentiment_mean, tweet_count, weighted_sentiment_mean (if weight_col given)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["_bucket"] = df[date_col].dt.floor("4h").dt.tz_localize(None)

    agg: dict = {
        "sentiment_mean":  (score_col, "mean"),
        "tweet_count":     (score_col, "count"),
    }
    if weight_col and weight_col in df.columns:
        def weighted_mean(grp):
            w = grp[weight_col]
            s = grp[score_col]
            denom = w.sum()
            return (s * w).sum() / denom if denom > 0 else s.mean()

        result = df.groupby("_bucket").apply(
            lambda g: pd.Series({
                "sentiment_mean":          g[score_col].mean(),
                "tweet_count":             len(g),
                "weighted_sentiment_mean": (g[score_col] * g[weight_col]).sum() / g[weight_col].sum()
                                           if g[weight_col].sum() > 0 else g[score_col].mean(),
            })
        )
    else:
        result = df.groupby("_bucket").agg(**agg)

    result.index.name = "bucket"
    return result


def save_4h_sentiment(agg: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the 4H sentiment aggregation DataFrame to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg = agg.copy()
    if "tweet_count" in agg.columns:
        agg["tweet_count"] = agg["tweet_count"].astype(int)
    agg.to_csv(out, float_format="%.4f")
    print(f"4H sentiment saved -> {out}  ({len(agg)} buckets)")
    return out


def load_4h_sentiment_series(
    csv_path: str | Path,
    use_weighted: bool = True,
) -> pd.Series:
    """Load a 4H sentiment CSV and return as a pd.Series indexed by bucket timestamp."""
    df = pd.read_csv(csv_path, index_col="bucket", parse_dates=True)

    if use_weighted and "weighted_sentiment_mean" in df.columns:
        series = df["weighted_sentiment_mean"]
    elif "sentiment_mean" in df.columns:
        series = df["sentiment_mean"]
    else:
        raise ValueError(f"No usable sentiment column in {csv_path}")

    series.name = "Sentiment"
    return series
