"""Download raw Bitcoin market data for the project pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


COINGECKO_MARKET_CHART_RANGE_URL = (
    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
)
DEFAULT_OUTPUT_PATH = Path("data/raw/bitcoin_price_raw.csv")
DEFAULT_LOOKBACK_DAYS = 365


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD format."
        ) from exc


def _date_to_unix_seconds(value: date, end_of_day: bool = False) -> int:
    day_time = time.max if end_of_day else time.min
    return int(datetime.combine(value, day_time, tzinfo=timezone.utc).timestamp())


def fetch_bitcoin_market_data(
    start_date: date,
    end_date: date,
    vs_currency: str = "usd",
) -> dict[str, Any]:
    """Fetch raw Bitcoin price, market cap, and volume data from CoinGecko."""
    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date")

    query = urlencode(
        {
            "vs_currency": vs_currency,
            "from": _date_to_unix_seconds(start_date),
            "to": _date_to_unix_seconds(end_date, end_of_day=True),
        }
    )
    url = f"{COINGECKO_MARKET_CHART_RANGE_URL}?{query}"
    request = Request(url, headers={"User-Agent": "bitirme-projesi/0.1"})

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"CoinGecko request failed: {exc.code} {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to CoinGecko: {exc.reason}") from exc


def build_price_dataframe(raw_data: dict[str, Any]) -> pd.DataFrame:
    """Convert CoinGecko market chart JSON into one raw CSV-friendly table."""
    prices = pd.DataFrame(raw_data.get("prices", []), columns=["timestamp_ms", "price_usd"])
    market_caps = pd.DataFrame(
        raw_data.get("market_caps", []), columns=["timestamp_ms", "market_cap_usd"]
    )
    volumes = pd.DataFrame(
        raw_data.get("total_volumes", []), columns=["timestamp_ms", "total_volume_usd"]
    )

    if prices.empty:
        raise ValueError("CoinGecko response did not contain price data")

    data = prices.merge(market_caps, on="timestamp_ms", how="left").merge(
        volumes, on="timestamp_ms", how="left"
    )
    data.insert(
        0,
        "date",
        pd.to_datetime(data["timestamp_ms"], unit="ms", utc=True).dt.date.astype(str),
    )
    return data[["date", "timestamp_ms", "price_usd", "market_cap_usd", "total_volume_usd"]]


def save_raw_data(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw Bitcoin market data and save it as CSV."
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=datetime.now(timezone.utc).date() - timedelta(days=DEFAULT_LOOKBACK_DAYS),
        help=f"Start date in YYYY-MM-DD format. Default: last {DEFAULT_LOOKBACK_DAYS} days.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=datetime.now(timezone.utc).date(),
        help="End date in YYYY-MM-DD format. Default: today in UTC.",
    )
    parser.add_argument(
        "--vs-currency",
        default="usd",
        help="Currency used by CoinGecko for Bitcoin prices. Default: usd.",
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
    raw_data = fetch_bitcoin_market_data(args.start_date, args.end_date, args.vs_currency)
    data = build_price_dataframe(raw_data)
    output_path = save_raw_data(data, args.output)
    print(f"Saved {len(data)} rows to {output_path}")


if __name__ == "__main__":
    main()
