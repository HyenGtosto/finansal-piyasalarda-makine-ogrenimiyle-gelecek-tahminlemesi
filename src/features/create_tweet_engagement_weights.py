"""Create a single tweet engagement weight from exposure and interaction counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/interim/bitcoin_tweets_cleaned.csv")
DEFAULT_OUTPUT_PATH = Path("data/interim/bitcoin_tweets_weighted.csv")
COUNT_COLUMNS = [
    "view_count",
    "like_count",
    "retweet_count",
    "reply_count",
    "quote_count",
    "bookmark_count",
    "author_followers",
]
OUTPUT_COLUMNS = [
    "tweet_id",
    "created_at",
    "text",
    "engagement_score",
    "engagement_weight",
]
COMPONENT_WEIGHTS = {
    "view_count": 0.50,
    "like_count": 0.20,
    "retweet_count": 0.10,
    "reply_count": 0.05,
    "quote_count": 0.05,
    "bookmark_count": 0.05,
    "author_followers": 0.05,
}
VERIFIED_MULTIPLIER = 1.05
MIN_ENGAGEMENT_WEIGHT = 0.25
MAX_ENGAGEMENT_WEIGHT = 5.0
CAP_QUANTILE = 0.99


def _log_capped_score(series: pd.Series, cap_quantile: float = CAP_QUANTILE) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    cap = numeric.quantile(cap_quantile)
    if cap <= 0:
        return pd.Series(0.0, index=series.index)

    capped = numeric.clip(upper=cap)
    return pd.Series(
        data=(capped + 1).apply("log") / pd.Series([cap + 1]).apply("log").iloc[0],
        index=series.index,
    ).clip(0, 1)


def add_engagement_weights(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in COUNT_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    weighted = data.copy()
    influence_score = pd.Series(0.0, index=weighted.index)

    for column, component_weight in COMPONENT_WEIGHTS.items():
        component = _log_capped_score(weighted[column])
        weighted[f"{column}_score"] = component
        influence_score += component * component_weight

    verified = weighted.get("author_verified", False)
    verified = verified.fillna(False).astype(bool) if isinstance(verified, pd.Series) else False
    influence_score = influence_score * (1 + (verified.astype(float) * (VERIFIED_MULTIPLIER - 1)))

    mean_score = influence_score.mean()
    if mean_score <= 0:
        weighted["engagement_weight"] = 1.0
    else:
        weighted["engagement_weight"] = (influence_score / mean_score).clip(
            MIN_ENGAGEMENT_WEIGHT,
            MAX_ENGAGEMENT_WEIGHT,
        )

    weighted["engagement_score"] = influence_score
    weighted["engagement_score"] = weighted["engagement_score"].round(4)
    weighted["engagement_weight"] = weighted["engagement_weight"].round(4)
    return weighted[OUTPUT_COLUMNS]


def create_weighted_tweet_dataset(input_path: Path, output_path: Path) -> Path:
    data = pd.read_csv(input_path)
    weighted = add_engagement_weights(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weighted.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create engagement weights for cleaned tweet data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Cleaned tweet CSV path. Default: {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Weighted tweet CSV path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = create_weighted_tweet_dataset(args.input, args.output)
    print(f"Saved weighted tweet data to {output_path}")


if __name__ == "__main__":
    main()
