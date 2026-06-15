"""Clean tweet text for sentiment analysis."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/interim/bitcoin_tweets_trimmed.csv")
DEFAULT_OUTPUT_PATH = Path("data/interim/bitcoin_tweets_cleaned.csv")

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@(\w+)")
HASHTAG_RE = re.compile(r"#(\w+)")
CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,15})\b")
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0000200D"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
WHITESPACE_RE = re.compile(r"\s+")


def clean_tweet_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = html.unescape(str(value))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(r"\1", text)
    text = text.replace("@", " ")
    text = HASHTAG_RE.sub(r"\1", text)
    text = text.replace("#", " ")
    text = CASHTAG_RE.sub(r"\1", text)
    text = EMOJI_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_tweet_dataset(input_path: Path, output_path: Path) -> Path:
    data = pd.read_csv(input_path)
    if "text" not in data.columns:
        raise ValueError("Input data must contain a text column")

    cleaned = data.copy()
    cleaned["text"] = cleaned["text"].map(clean_tweet_text)
    cleaned = cleaned[cleaned["text"].str.len() > 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean interim tweet text for sentiment analysis."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Trimmed tweet CSV path. Default: {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Cleaned tweet CSV path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = clean_tweet_dataset(args.input, args.output)
    print(f"Saved cleaned tweet data to {output_path}")


if __name__ == "__main__":
    main()
