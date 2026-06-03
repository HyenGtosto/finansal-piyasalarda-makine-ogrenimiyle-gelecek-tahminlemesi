"""Download raw Bitcoin-related tweets for later text cleaning and sentiment analysis."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


X_FULL_ARCHIVE_SEARCH_URL = "https://api.x.com/2/tweets/search/all"
DEFAULT_PRICE_INPUT_PATH = Path("data/raw/bitcoin_price_raw.csv")
DEFAULT_OUTPUT_PATH = Path("data/raw/bitcoin_tweets_raw.csv")
DEFAULT_QUERY = (
    '("bitcoin" OR btc OR "crypto market" OR "crypto markets" OR cryptocurrency) '
    "lang:en -is:retweet"
)
DEFAULT_TWEET_FIELDS = (
    "id,text,created_at,author_id,lang,public_metrics,possibly_sensitive,"
    "conversation_id,referenced_tweets"
)
DEFAULT_MAX_RESULTS = 500


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD format."
        ) from exc


def _to_rfc3339(value: date) -> str:
    timestamp = datetime.combine(value, time.min, tzinfo=timezone.utc)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def read_price_date_range(price_input_path: Path) -> tuple[date, date]:
    prices = pd.read_csv(price_input_path, usecols=["date"])
    if prices.empty:
        raise ValueError(f"No dates found in {price_input_path}")

    dates = pd.to_datetime(prices["date"], utc=True).dt.date
    return dates.min(), dates.max()


def get_bearer_token() -> str:
    token = os.getenv("TWITTER_BEARER_TOKEN") or os.getenv("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError(
            "Set TWITTER_BEARER_TOKEN or X_BEARER_TOKEN before downloading tweets."
        )
    return token


def fetch_tweets(
    query: str,
    start_date: date,
    end_date: date,
    bearer_token: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_tweets: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch matching tweets from X full-archive search across the date range."""
    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date")
    if not 10 <= max_results <= 500:
        raise ValueError("max_results must be between 10 and 500")

    tweets: list[dict[str, Any]] = []
    next_token: str | None = None
    exclusive_end_date = end_date + timedelta(days=1)

    while True:
        params = {
            "query": query,
            "start_time": _to_rfc3339(start_date),
            "end_time": _to_rfc3339(exclusive_end_date),
            "max_results": max_results,
            "tweet.fields": DEFAULT_TWEET_FIELDS,
        }
        if next_token:
            params["next_token"] = next_token

        request = Request(
            f"{X_FULL_ARCHIVE_SEARCH_URL}?{urlencode(params)}",
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": "bitirme-projesi/0.1",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"X API request failed: {exc.code} {message}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not connect to X API: {exc.reason}") from exc

        tweets.extend(payload.get("data", []))
        if max_tweets and len(tweets) >= max_tweets:
            return tweets[:max_tweets]

        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            return tweets


def build_tweet_dataframe(tweets: list[dict[str, Any]], query: str) -> pd.DataFrame:
    columns = [
        "tweet_id",
        "created_at",
        "date",
        "author_id",
        "lang",
        "text",
        "retweet_count",
        "reply_count",
        "like_count",
        "quote_count",
        "bookmark_count",
        "impression_count",
        "possibly_sensitive",
        "conversation_id",
        "referenced_tweets",
        "query",
    ]

    rows: list[dict[str, Any]] = []
    for tweet in tweets:
        metrics = tweet.get("public_metrics", {})
        created_at = tweet.get("created_at", "")
        rows.append(
            {
                "tweet_id": tweet.get("id"),
                "created_at": created_at,
                "date": pd.to_datetime(created_at, utc=True).date().isoformat()
                if created_at
                else "",
                "author_id": tweet.get("author_id"),
                "lang": tweet.get("lang"),
                "text": tweet.get("text", ""),
                "retweet_count": metrics.get("retweet_count", 0),
                "reply_count": metrics.get("reply_count", 0),
                "like_count": metrics.get("like_count", 0),
                "quote_count": metrics.get("quote_count", 0),
                "bookmark_count": metrics.get("bookmark_count", 0),
                "impression_count": metrics.get("impression_count", 0),
                "possibly_sensitive": tweet.get("possibly_sensitive"),
                "conversation_id": tweet.get("conversation_id"),
                "referenced_tweets": json.dumps(tweet.get("referenced_tweets", [])),
                "query": query,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def save_raw_data(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw Bitcoin-related tweets for the Bitcoin price date range."
    )
    parser.add_argument(
        "--price-input",
        type=Path,
        default=DEFAULT_PRICE_INPUT_PATH,
        help=f"Bitcoin price CSV used to infer the date range. Default: {DEFAULT_PRICE_INPUT_PATH}.",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        help="Optional start date in YYYY-MM-DD format. Defaults to price CSV minimum date.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        help="Optional end date in YYYY-MM-DD format. Defaults to price CSV maximum date.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="X API search query used to collect raw tweets.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Tweets per API page, from 10 to 500. Default: 500.",
    )
    parser.add_argument(
        "--max-tweets",
        type=int,
        help="Optional cap for testing or limited API plans. Omit to paginate all accessible results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_start_date, price_end_date = read_price_date_range(args.price_input)
    start_date = args.start_date or price_start_date
    end_date = args.end_date or price_end_date

    tweets = fetch_tweets(
        query=args.query,
        start_date=start_date,
        end_date=end_date,
        bearer_token=get_bearer_token(),
        max_results=args.max_results,
        max_tweets=args.max_tweets,
    )
    data = build_tweet_dataframe(tweets, args.query)
    output_path = save_raw_data(data, args.output)
    print(f"Saved {len(data)} raw tweets to {output_path}")


if __name__ == "__main__":
    main()
