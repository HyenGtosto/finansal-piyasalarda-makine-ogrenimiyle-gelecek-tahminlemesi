"""Build the final 4-hour model dataset from price and tweet sentiment features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PRICE_INPUT_PATH = Path("data/raw/bitcoin_price_raw.csv")
DEFAULT_SENTIMENT_INPUT_PATH = Path("data/interim/bitcoin_tweets_sentiment.csv")
DEFAULT_WEIGHT_INPUT_PATH = Path("data/interim/bitcoin_tweets_weighted.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/final_dataset.csv")
DEFAULT_WINDOW_HOURS = 4
SENTIMENT_COLUMNS = ["tweet_id", "created_at", "sentiment_score"]
OPTIONAL_SENTIMENT_COLUMNS = ["sentiment_label", "engagement_weight"]
DROP_FINAL_COLUMNS = ["price_points"]
TARGET_COLUMNS = ["next_price_close", "next_4h_return", "target_up_next_4h"]


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
    features["weighted_sentiment_mean"] = features["weighted_sentiment_mean"].replace(
        [float("inf"), float("-inf")],
        0,
    )
    return features


def add_leakage_safe_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.sort_values("window_start").reset_index(drop=True).copy()

    features["weighted_sentiment_delta_4h"] = (
        features["weighted_sentiment_mean"] - features["weighted_sentiment_mean"].shift(1)
    )
    features["sentiment_mean_delta_4h"] = (
        features["sentiment_mean"] - features["sentiment_mean"].shift(1)
    )
    features["tweet_count_delta_4h"] = features["tweet_count"] - features["tweet_count"].shift(1)
    features["engagement_weight_sum_delta_4h"] = (
        features["engagement_weight_sum"] - features["engagement_weight_sum"].shift(1)
    )

    add_rolling_features(features)
    add_zscore_features(features)
    cleanup_derived_feature_values(features)
    return features


def add_rolling_features(features: pd.DataFrame) -> None:
    six_windows = 6
    forty_two_windows = 42

    features["weighted_sentiment_mean_24h"] = (
        features["weighted_sentiment_mean"].rolling(six_windows, min_periods=1).mean()
    )
    features["weighted_sentiment_std_24h"] = (
        features["weighted_sentiment_mean"].rolling(six_windows, min_periods=2).std()
    )
    features["tweet_count_mean_24h"] = (
        features["tweet_count"].rolling(six_windows, min_periods=1).mean()
    )
    features["tweet_count_std_24h"] = (
        features["tweet_count"].rolling(six_windows, min_periods=2).std()
    )
    features["engagement_weight_sum_mean_24h"] = (
        features["engagement_weight_sum"].rolling(six_windows, min_periods=1).mean()
    )
    features["engagement_weight_sum_std_24h"] = (
        features["engagement_weight_sum"].rolling(six_windows, min_periods=2).std()
    )

    features["weighted_sentiment_mean_7d"] = (
        features["weighted_sentiment_mean"].rolling(forty_two_windows, min_periods=1).mean()
    )
    features["tweet_count_mean_7d"] = (
        features["tweet_count"].rolling(forty_two_windows, min_periods=1).mean()
    )
    features["engagement_weight_sum_mean_7d"] = (
        features["engagement_weight_sum"].rolling(forty_two_windows, min_periods=1).mean()
    )
    features["price_return_mean_24h"] = (
        features["price_return"].rolling(six_windows, min_periods=1).mean()
    )
    features["price_return_std_24h"] = (
        features["price_return"].rolling(six_windows, min_periods=2).std()
    )
    features["volume_sum_mean_24h"] = (
        features["volume_sum"].rolling(six_windows, min_periods=1).mean()
    )
    features["_volume_sum_std_24h"] = (
        features["volume_sum"].rolling(six_windows, min_periods=2).std()
    )


def add_zscore_features(features: pd.DataFrame) -> None:
    features["tweet_count_zscore_24h"] = (
        features["tweet_count"] - features["tweet_count_mean_24h"]
    ) / features["tweet_count_std_24h"]
    features["engagement_weight_sum_zscore_24h"] = (
        features["engagement_weight_sum"] - features["engagement_weight_sum_mean_24h"]
    ) / features["engagement_weight_sum_std_24h"]
    features["weighted_sentiment_zscore_24h"] = (
        features["weighted_sentiment_mean"] - features["weighted_sentiment_mean_24h"]
    ) / features["weighted_sentiment_std_24h"]
    features["volume_sum_zscore_24h"] = (
        features["volume_sum"] - features["volume_sum_mean_24h"]
    ) / features["_volume_sum_std_24h"]
    features.drop(columns=["_volume_sum_std_24h"], inplace=True)


def cleanup_derived_feature_values(features: pd.DataFrame) -> None:
    derived_columns = [
        "weighted_sentiment_delta_4h",
        "sentiment_mean_delta_4h",
        "tweet_count_delta_4h",
        "engagement_weight_sum_delta_4h",
        "weighted_sentiment_std_24h",
        "tweet_count_std_24h",
        "engagement_weight_sum_std_24h",
        "price_return_std_24h",
        "tweet_count_zscore_24h",
        "engagement_weight_sum_zscore_24h",
        "weighted_sentiment_zscore_24h",
        "volume_sum_zscore_24h",
    ]
    existing_columns = [column for column in derived_columns if column in features.columns]
    features[existing_columns] = (
        features[existing_columns]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )


def get_model_feature_columns(data: pd.DataFrame) -> list[str]:
    excluded_columns = set(TARGET_COLUMNS + ["window_start"])
    return [column for column in data.columns if column not in excluded_columns]


def validate_final_dataset(data: pd.DataFrame) -> None:
    if not data["window_start"].is_monotonic_increasing:
        raise ValueError("window_start must be sorted ascending")
    if data["window_start"].duplicated().any():
        raise ValueError("window_start contains duplicate values")

    feature_columns = get_model_feature_columns(data)
    leaked_targets = sorted(set(feature_columns).intersection(TARGET_COLUMNS))
    if leaked_targets:
        raise ValueError(f"Target columns are present in feature columns: {leaked_targets}")

    numeric_data = data.select_dtypes(include="number")
    if numeric_data.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Dataset contains infinite values")
    if data.isna().any().any():
        missing_counts = data.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Dataset contains NaN values: {missing_counts}")

    class_balance = data["target_up_next_4h"].value_counts(normalize=True).sort_index()
    print("Final dataset validation")
    print(f"Rows: {len(data)}")
    print(f"Date range: {data['window_start'].iloc[0]} to {data['window_start'].iloc[-1]}")
    print("target_up_next_4h class balance:")
    for target_class, ratio in class_balance.items():
        print(f"  {target_class}: {ratio:.4f}")


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
    final = final.dropna(subset=["next_price_close", "next_4h_return"])
    final = add_leakage_safe_features(final)
    final = final.drop(columns=[column for column in DROP_FINAL_COLUMNS if column in final.columns])
    final["window_start"] = final["window_start"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    validate_final_dataset(final)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    return output_path


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
