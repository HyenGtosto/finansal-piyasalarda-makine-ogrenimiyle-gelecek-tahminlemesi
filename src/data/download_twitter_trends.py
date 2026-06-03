"""Download raw daily Twitter/X trend counts for Bitcoin-related discussion."""

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


X_FULL_ARCHIVE_COUNTS_URL = "https://api.x.com/2/tweets/counts/all"
DEFAULT_PRICE_INPUT_PATH = Path("data/raw/bitcoin_price_raw.csv")
DEFAULT_OUTPUT_PATH = Path("data/raw/bitcoin_twitter_trends_raw.csv")
DEFAULT_QUERY = (
    '("bitcoin" OR btc OR "crypto market" OR "crypto markets" OR cryptocurrency) '
    "lang:en -is:retweet"
)


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
            "Set TWITTER_BEARER_TOKEN or X_BEARER_TOKEN before downloading X trend data."
        )
    return token


def fetch_daily_tweet_counts(
    query: str,
    start_date: date,
    end_date: date,
    bearer_token: str,
) -> list[dict[str, Any]]:
    """Fetch daily matching post counts from the X full-archive counts endpoint."""
    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date")

    rows: list[dict[str, Any]] = []
    next_token: str | None = None
    exclusive_end_date = end_date + timedelta(days=1)

    while True:
        params = {
            "query": query,
            "start_time": _to_rfc3339(start_date),
            "end_time": _to_rfc3339(exclusive_end_date),
            "granularity": "day",
        }
        if next_token:
            params["next_token"] = next_token

        request = Request(
            f"{X_FULL_ARCHIVE_COUNTS_URL}?{urlencode(params)}",
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

        rows.extend(payload.get("data", []))
        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            return rows


def build_trend_dataframe(
    rows: list[dict[str, Any]],
    query: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    all_dates = pd.DataFrame(
        {
            "date": pd.date_range(start_date, end_date, freq="D")
            .date.astype(str)
            .tolist()
        }
    )

    if rows:
        counts = pd.DataFrame(rows)
        counts["date"] = pd.to_datetime(counts["start"], utc=True).dt.date.astype(str)
        counts = counts.rename(
            columns={"start": "start_time", "end": "end_time"}
        )[["date", "start_time", "end_time", "tweet_count"]]
    else:
        counts = pd.DataFrame(columns=["date", "start_time", "end_time", "tweet_count"])

    data = all_dates.merge(counts, on="date", how="left")
    data["tweet_count"] = (
        pd.to_numeric(data["tweet_count"], errors="coerce").fillna(0).astype(int)
    )
    data["start_time"] = data["start_time"].fillna(
        pd.to_datetime(data["date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    )
    data["end_time"] = data["end_time"].fillna(
        (
            pd.to_datetime(data["date"], utc=True) + pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    )
    data["query"] = query
    return data[["date", "start_time", "end_time", "tweet_count", "query"]]


def save_raw_data(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily Twitter/X trend counts for Bitcoin-related posts."
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
        help="X API search query used for trend counts.",
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

    rows = fetch_daily_tweet_counts(
        query=args.query,
        start_date=start_date,
        end_date=end_date,
        bearer_token=get_bearer_token(),
    )
    data = build_trend_dataframe(rows, args.query, start_date, end_date)
    output_path = save_raw_data(data, args.output)
    print(f"Saved {len(data)} daily Twitter/X trend rows to {output_path}")


if __name__ == "__main__":
    main()
