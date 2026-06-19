"""Build the final 4-hour model dataset from price and tweet sentiment features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PRICE_INPUT_PATH = Path("data/raw/bitcoin_price_raw.csv")
DEFAULT_SENTIMENT_INPUT_PATH = Path("data/interim/bitcoin_tweets_sentiment.csv")
DEFAULT_WEIGHT_INPUT_PATH = Path("data/interim/bitcoin_tweets_weighted.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/final_dataset.csv")
DEFAULT_WINDOW_HOURS = 4
SENTIMENT_COLUMNS = ["tweet_id", "created_at", "sentiment_score"]
OPTIONAL_SENTIMENT_COLUMNS = ["sentiment_label", "engagement_weight"]
DROP_FINAL_COLUMNS = ["price_points"]
TARGET_COLUMNS = {"next_price_close", "next_4h_return", "target_up_next_4h"}
TIME_COLUMN = "window_start"
ROUNDING_RULES = {
    "price_open": 2,
    "price_high": 2,
    "price_low": 2,
    "price_close": 2,
    "price_mean": 2,
    "next_price_close": 2,
    "market_cap_mean": 0,
    "volume_sum": 0,
    "price_return": 6,
    "price_range_pct": 6,
    "next_4h_return": 6,
    "tweet_count": 0,
    "sentiment_mean": 4,
    "sentiment_min": 4,
    "sentiment_max": 4,
    "sentiment_std": 4,
    "positive_tweet_ratio": 4,
    "negative_tweet_ratio": 4,
    "engagement_weight_mean": 4,
    "engagement_weight_sum": 4,
    "weighted_sentiment_sum": 4,
    "weighted_sentiment_mean": 4,
}


def add_window_start(
    data: pd.DataFrame,
    datetime_column: str,
    window_hours: int,
) -> pd.DataFrame:
    windowed = data.copy()
    windowed[datetime_column] = pd.to_datetime(windowed[datetime_column], utc=True)
    windowed["window_start"] = windowed[datetime_column].dt.floor(f"{window_hours}h")
    return windowed


def build_price_features(price_input_path: Path, window_hours: int) -> pd.DataFrame:
    prices = pd.read_csv(price_input_path)
    required_columns = ["datetime_utc", "price_usd", "market_cap_usd", "total_volume_usd"]
    missing_columns = [column for column in required_columns if column not in prices.columns]
    if missing_columns:
        raise ValueError(f"Missing price columns: {missing_columns}")

    prices = add_window_start(prices, "datetime_utc", window_hours)
    grouped = prices.groupby("window_start", as_index=False)
    features = grouped.agg(
        price_open=("price_usd", "first"),
        price_high=("price_usd", "max"),
        price_low=("price_usd", "min"),
        price_close=("price_usd", "last"),
        price_mean=("price_usd", "mean"),
        market_cap_mean=("market_cap_usd", "mean"),
        volume_sum=("total_volume_usd", "sum"),
        price_points=("price_usd", "count"),
    )
    features["price_return"] = (
        features["price_close"] - features["price_open"]
    ) / features["price_open"]
    features["price_range_pct"] = (
        features["price_high"] - features["price_low"]
    ) / features["price_open"]
    features["next_price_close"] = features["price_close"].shift(-1)
    features["next_4h_return"] = (
        features["next_price_close"] - features["price_close"]
    ) / features["price_close"]
    features["target_up_next_4h"] = (features["next_4h_return"] > 0).astype(int)
    return features


def read_sentiment_input(
    sentiment_input_path: Path,
    weight_input_path: Path,
) -> pd.DataFrame:
    sentiment = pd.read_csv(sentiment_input_path)
    missing_columns = [column for column in SENTIMENT_COLUMNS if column not in sentiment.columns]
    if missing_columns:
        raise ValueError(f"Missing sentiment columns: {missing_columns}")

    if "engagement_weight" not in sentiment.columns:
        weights = pd.read_csv(weight_input_path, usecols=["tweet_id", "engagement_weight"])
        sentiment = sentiment.merge(weights, on="tweet_id", how="left")

    sentiment["engagement_weight"] = (
        pd.to_numeric(sentiment["engagement_weight"], errors="coerce").fillna(1.0)
    )
    sentiment["sentiment_score"] = pd.to_numeric(
        sentiment["sentiment_score"], errors="coerce"
    )
    sentiment = sentiment.dropna(subset=["sentiment_score"])
    return sentiment


def build_sentiment_features(
    sentiment_input_path: Path,
    weight_input_path: Path,
    window_hours: int,
) -> pd.DataFrame:
    sentiment = read_sentiment_input(sentiment_input_path, weight_input_path)
    sentiment = add_window_start(sentiment, "created_at", window_hours)
    sentiment["positive_tweet"] = (sentiment["sentiment_score"] > 0).astype(int)
    sentiment["negative_tweet"] = (sentiment["sentiment_score"] < 0).astype(int)
    sentiment["weighted_sentiment_component"] = (
        sentiment["sentiment_score"] * sentiment["engagement_weight"]
    )

    grouped = sentiment.groupby("window_start", as_index=False)
    features = grouped.agg(
        tweet_count=("tweet_id", "count"),
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_min=("sentiment_score", "min"),
        sentiment_max=("sentiment_score", "max"),
        sentiment_std=("sentiment_score", "std"),
        positive_tweet_ratio=("positive_tweet", "mean"),
        negative_tweet_ratio=("negative_tweet", "mean"),
        engagement_weight_mean=("engagement_weight", "mean"),
        engagement_weight_sum=("engagement_weight", "sum"),
        weighted_sentiment_sum=("weighted_sentiment_component", "sum"),
    )
    features["weighted_sentiment_mean"] = (
        features["weighted_sentiment_sum"] / features["engagement_weight_sum"]
    )
    return features


def add_leakage_safe_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add historical features using only current and previous rows."""
    featured = data.copy()
    featured = featured.sort_values(TIME_COLUMN).reset_index(drop=True)

    featured["weighted_sentiment_delta_4h"] = (
        featured["weighted_sentiment_mean"] - featured["weighted_sentiment_mean"].shift(1)
    )
    featured["sentiment_mean_delta_4h"] = (
        featured["sentiment_mean"] - featured["sentiment_mean"].shift(1)
    )
    featured["tweet_count_delta_4h"] = featured["tweet_count"] - featured["tweet_count"].shift(1)
    featured["engagement_weight_sum_delta_4h"] = (
        featured["engagement_weight_sum"] - featured["engagement_weight_sum"].shift(1)
    )

    mean_24h_columns = {
        "weighted_sentiment_mean": "weighted_sentiment_mean_24h",
        "tweet_count": "tweet_count_mean_24h",
        "engagement_weight_sum": "engagement_weight_sum_mean_24h",
        "price_return": "price_return_mean_24h",
        "volume_sum": "volume_sum_mean_24h",
    }
    for source_column, output_column in mean_24h_columns.items():
        featured[output_column] = featured[source_column].rolling(window=6, min_periods=1).mean()

    std_24h_columns = {
        "weighted_sentiment_mean": "weighted_sentiment_std_24h",
        "tweet_count": "tweet_count_std_24h",
        "engagement_weight_sum": "engagement_weight_sum_std_24h",
        "price_return": "price_return_std_24h",
        "volume_sum": "volume_sum_std_24h",
    }
    for source_column, output_column in std_24h_columns.items():
        featured[output_column] = featured[source_column].rolling(window=6, min_periods=2).std()

    featured["tweet_count_zscore_24h"] = (
        (featured["tweet_count"] - featured["tweet_count_mean_24h"])
        / featured["tweet_count_std_24h"]
    )
    featured["engagement_weight_sum_zscore_24h"] = (
        (featured["engagement_weight_sum"] - featured["engagement_weight_sum_mean_24h"])
        / featured["engagement_weight_sum_std_24h"]
    )
    featured["weighted_sentiment_zscore_24h"] = (
        (featured["weighted_sentiment_mean"] - featured["weighted_sentiment_mean_24h"])
        / featured["weighted_sentiment_std_24h"]
    )
    featured["volume_sum_zscore_24h"] = (
        (featured["volume_sum"] - featured["volume_sum_mean_24h"])
        / featured["volume_sum_std_24h"]
    )

    featured["weighted_sentiment_mean_7d"] = (
        featured["weighted_sentiment_mean"].rolling(window=42, min_periods=1).mean()
    )
    featured["tweet_count_mean_7d"] = (
        featured["tweet_count"].rolling(window=42, min_periods=1).mean()
    )
    featured["engagement_weight_sum_mean_7d"] = (
        featured["engagement_weight_sum"].rolling(window=42, min_periods=1).mean()
    )

    cleanup_columns = [
        "weighted_sentiment_delta_4h",
        "sentiment_mean_delta_4h",
        "tweet_count_delta_4h",
        "engagement_weight_sum_delta_4h",
        "weighted_sentiment_std_24h",
        "tweet_count_std_24h",
        "engagement_weight_sum_std_24h",
        "price_return_std_24h",
        "volume_sum_std_24h",
        "tweet_count_zscore_24h",
        "engagement_weight_sum_zscore_24h",
        "weighted_sentiment_zscore_24h",
        "volume_sum_zscore_24h",
    ]
    featured[cleanup_columns] = (
        featured[cleanup_columns].replace([np.inf, -np.inf], 0).fillna(0)
    )
    return featured


def get_model_feature_columns(data: pd.DataFrame) -> list[str]:
    excluded_columns = TARGET_COLUMNS.union({TIME_COLUMN})
    return [
        column
        for column in data.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(data[column])
    ]


def validate_final_dataset(data: pd.DataFrame) -> None:
    if not data[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError("window_start must be sorted ascending")
    if data[TIME_COLUMN].duplicated().any():
        raise ValueError("window_start contains duplicate values")

    feature_columns = get_model_feature_columns(data)
    leaked_targets = sorted(set(feature_columns).intersection(TARGET_COLUMNS))
    if leaked_targets:
        raise ValueError(f"Target columns cannot be used as features: {leaked_targets}")

    numeric = data.select_dtypes(include=["number"])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Output dataset contains infinite values")
    if data.isna().any().any():
        raise ValueError("Output dataset contains NaN values")

    class_balance = data["target_up_next_4h"].value_counts(normalize=True).sort_index().to_dict()
    print(f"Rows: {len(data)}")
    print(f"Date range: {data[TIME_COLUMN].min()} to {data[TIME_COLUMN].max()}")
    print(f"target_up_next_4h balance: {class_balance}")


def build_final_dataset(
    price_input_path: Path,
    sentiment_input_path: Path,
    weight_input_path: Path,
    output_path: Path,
    window_hours: int,
) -> Path:
    price_features = build_price_features(price_input_path, window_hours)
    sentiment_features = build_sentiment_features(
        sentiment_input_path,
        weight_input_path,
        window_hours,
    )

    final = price_features.merge(sentiment_features, on="window_start", how="left")
    fill_zero_columns = [
        "tweet_count",
        "sentiment_mean",
        "sentiment_min",
        "sentiment_max",
        "sentiment_std",
        "positive_tweet_ratio",
        "negative_tweet_ratio",
        "engagement_weight_mean",
        "engagement_weight_sum",
        "weighted_sentiment_sum",
        "weighted_sentiment_mean",
    ]
    final[fill_zero_columns] = final[fill_zero_columns].fillna(0)
    final = add_leakage_safe_features(final)
    final = final.dropna(subset=["next_price_close", "next_4h_return"])
    final["window_start"] = final["window_start"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    final = final.drop(columns=[column for column in DROP_FINAL_COLUMNS if column in final.columns])
    final = round_final_dataset(final)
    validate_final_dataset(final)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    return output_path


def round_final_dataset(data: pd.DataFrame) -> pd.DataFrame:
    rounded = data.copy()
    for column, decimals in ROUNDING_RULES.items():
        if column in rounded.columns:
            rounded[column] = pd.to_numeric(rounded[column], errors="coerce").round(decimals)

    if "tweet_count" in rounded.columns:
        rounded["tweet_count"] = rounded["tweet_count"].astype(int)
    return rounded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final 4-hour model training dataset."
    )
    parser.add_argument(
        "--price-input",
        type=Path,
        default=DEFAULT_PRICE_INPUT_PATH,
        help=f"Hourly Bitcoin price CSV path. Default: {DEFAULT_PRICE_INPUT_PATH}.",
    )
    parser.add_argument(
        "--sentiment-input",
        type=Path,
        default=DEFAULT_SENTIMENT_INPUT_PATH,
        help=f"Tweet sentiment CSV path. Default: {DEFAULT_SENTIMENT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--weight-input",
        type=Path,
        default=DEFAULT_WEIGHT_INPUT_PATH,
        help=f"Weighted tweet CSV path. Default: {DEFAULT_WEIGHT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Final dataset output path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=DEFAULT_WINDOW_HOURS,
        help=f"Aggregation window in hours. Default: {DEFAULT_WINDOW_HOURS}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_final_dataset(
        args.price_input,
        args.sentiment_input,
        args.weight_input,
        args.output,
        args.window_hours,
    )
    print(f"Saved final model dataset to {output_path}")


if __name__ == "__main__":
    main()
