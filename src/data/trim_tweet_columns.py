"""Create an interim tweet dataset with only sentiment and weighting inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/raw/bitcoin_getxapi_tweets_raw.csv")
DEFAULT_OUTPUT_PATH = Path("data/interim/bitcoin_tweets_trimmed.csv")
KEEP_COLUMNS = [
    "tweet_id",
    "created_at",
    "text",
    "author_followers",
    "author_verified",
    "retweet_count",
    "reply_count",
    "like_count",
    "quote_count",
    "bookmark_count",
    "view_count",
]


def trim_tweet_columns(input_path: Path, output_path: Path) -> Path:
    data = pd.read_csv(input_path)
    missing_columns = [column for column in KEEP_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    trimmed = data[KEEP_COLUMNS].copy()
    trimmed = trimmed.drop_duplicates(subset=["tweet_id"])
    trimmed = trimmed.sort_values("created_at")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim raw GetXAPI tweets into an interim sentiment dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Raw tweet CSV path. Default: {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Interim output CSV path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = trim_tweet_columns(args.input, args.output)
    print(f"Saved trimmed tweet data to {output_path}")


if __name__ == "__main__":
    main()
