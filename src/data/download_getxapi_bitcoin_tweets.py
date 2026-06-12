"""Download filtered Bitcoin tweet data from GetXAPI for sentiment analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


GETXAPI_ADVANCED_SEARCH_URL = "https://api.getxapi.com/twitter/tweet/advanced_search"
DEFAULT_PRICE_INPUT_PATH = Path("data/raw/bitcoin_price_raw.csv")
DEFAULT_OUTPUT_PATH = Path("data/raw/bitcoin_getxapi_tweets_raw.csv")
DEFAULT_PRODUCT = "Latest"
DEFAULT_TWEETS_PER_DAY = 400
DEFAULT_CHUNK_HOURS = 4
DEFAULT_MAX_CALLS_PER_WINDOW = 1
DEFAULT_MAX_WINDOWS_PER_RUN = 0
DEFAULT_MAX_DAYS_PER_RUN = 1
DEFAULT_SAMPLE_WINDOW_MINUTES = 60
DEFAULT_RANDOM_SEED = 42
RESULTS_PER_CALL = 20
GETXAPI_COST_PER_CALL_USD = 0.001
ENV_FILE_PATH = Path(".env")
PROGRESS_COLUMNS = [
    "query",
    "product",
    "window_start",
    "window_end",
    "tweet_count",
    "api_calls",
    "status",
    "message",
]

BASE_QUERY = "bitcoin"
SENTIMENT_QUERY = (
    "(buy OR sell OR bullish OR bearish OR pump OR dump OR crash OR rally "
    "OR moon OR fear OR greed)"
)
NOISE_REDUCTION_QUERY = "lang:en -filter:retweets -filter:replies"
ENGAGEMENT_QUERY = "min_faves:2"
DEFAULT_QUERY = f"{BASE_QUERY} {SENTIMENT_QUERY} {NOISE_REDUCTION_QUERY}"


@dataclass(frozen=True)
class FetchResult:
    tweets: list[dict[str, Any]]
    api_calls: int


@dataclass(frozen=True)
class DownloadSummary:
    tweet_count: int
    api_calls: int
    stopped_reason: str | None = None


class GetXAPIRequestError(RuntimeError):
    def __init__(self, message: str, partial_result: FetchResult) -> None:
        super().__init__(message)
        self.partial_result = partial_result


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD format."
        ) from exc


def _parse_twitter_datetime(value: str | None) -> str:
    if not value:
        return ""

    try:
        parsed = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return ""
        return parsed.isoformat()

    return parsed.astimezone(timezone.utc).isoformat()


def _date_from_created_at(value: str | None) -> str:
    parsed = _parse_twitter_datetime(value)
    if not parsed:
        return ""
    return pd.to_datetime(parsed, utc=True).date().isoformat()


def normalize_tweet_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def read_price_date_range(price_input_path: Path) -> tuple[date, date]:
    prices = pd.read_csv(price_input_path, usecols=["date"])
    if prices.empty:
        raise ValueError(f"No dates found in {price_input_path}")

    dates = pd.to_datetime(prices["date"], utc=True).dt.date
    return dates.min(), dates.max()


def load_env_file(env_file_path: Path = ENV_FILE_PATH) -> None:
    if not env_file_path.exists():
        return

    for line in env_file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str:
    load_env_file()
    api_key = os.getenv("GETXAPI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GETXAPI_API_KEY before downloading GetXAPI tweet data.")
    return api_key


def iter_dates(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date")

    days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=offset) for offset in range(days)]


def build_daily_query(query: str, day: date) -> str:
    next_day = day + timedelta(days=1)
    return f"{query} since:{day.isoformat()} until:{next_day.isoformat()}"


def build_time_window_query(
    query: str,
    start_datetime: datetime,
    end_datetime: datetime,
) -> str:
    start_timestamp = int(start_datetime.timestamp())
    end_timestamp = int(end_datetime.timestamp())
    return f"{query} since_time:{start_timestamp} until_time:{end_timestamp}"


def choose_sample_window(
    window_start: datetime,
    window_end: datetime,
    sample_window_minutes: int | None,
    random_seed: int,
) -> tuple[datetime, datetime]:
    if sample_window_minutes is None:
        return window_start, window_end

    full_window_seconds = int((window_end - window_start).total_seconds())
    sample_window_seconds = sample_window_minutes * 60
    if sample_window_seconds >= full_window_seconds:
        return window_start, window_end
    if sample_window_seconds < 60:
        raise ValueError("sample_window_minutes must be at least 1")

    key = f"{random_seed}:{window_start.isoformat()}:{window_end.isoformat()}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    max_offset = full_window_seconds - sample_window_seconds
    offset_seconds = int(digest[:12], 16) % (max_offset + 1)
    sample_start = window_start + timedelta(seconds=offset_seconds)
    return sample_start, sample_start + timedelta(seconds=sample_window_seconds)


def iter_time_windows(
    start_date: date,
    end_date: date,
    chunk_hours: int,
) -> list[tuple[date, datetime, datetime]]:
    if chunk_hours < 1 or chunk_hours > 24:
        raise ValueError("chunk_hours must be between 1 and 24")

    windows: list[tuple[date, datetime, datetime]] = []
    for day in iter_dates(start_date, end_date):
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        window_start = day_start
        while window_start < day_end:
            window_end = min(window_start + timedelta(hours=chunk_hours), day_end)
            windows.append((day, window_start, window_end))
            window_start = window_end

    return windows


def tweets_per_window(tweets_per_day: int, chunk_hours: int) -> int:
    chunks_per_day = 24 // chunk_hours
    if 24 % chunk_hours:
        chunks_per_day += 1
    return max(1, -(-tweets_per_day // chunks_per_day))


def select_incomplete_days(
    windows: list[tuple[date, datetime, datetime]],
    completed_windows: set[tuple[str, str, str, str]],
    query: str,
    product: str,
    sample_window_minutes: int | None,
    random_seed: int,
    max_days: int | None,
) -> set[date] | None:
    if max_days is None:
        return None

    selected_days: list[date] = []
    selected_set: set[date] = set()
    for day, window_start, window_end in windows:
        sample_start, sample_end = choose_sample_window(
            window_start,
            window_end,
            sample_window_minutes,
            random_seed,
        )
        completed_key = window_key(query, product, sample_start, sample_end)
        if completed_key in completed_windows or day in selected_set:
            continue

        selected_days.append(day)
        selected_set.add(day)
        if len(selected_days) >= max_days:
            break

    return selected_set


def build_query(
    include_sentiment_terms: bool = False,
    exclude_links: bool = False,
    min_faves: int | None = None,
) -> str:
    parts = [BASE_QUERY]
    if include_sentiment_terms:
        parts.append(SENTIMENT_QUERY)

    parts.append(NOISE_REDUCTION_QUERY)
    if exclude_links:
        parts.append("-filter:links")
    if min_faves is not None and min_faves > 0:
        parts.append(f"min_faves:{min_faves}")

    return " ".join(parts)


def fetch_tweets_for_query(
    query: str,
    api_key: str,
    product: str,
    max_tweets: int,
    max_api_calls: int | None = None,
    debug: bool = False,
) -> FetchResult:
    tweets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    cursor: str | None = None
    api_calls = 0

    while len(tweets) < max_tweets:
        if max_api_calls is not None and api_calls >= max_api_calls:
            break

        params = {"q": query, "product": product}
        if cursor:
            params["cursor"] = cursor

        request = Request(
            f"{GETXAPI_ADVANCED_SEARCH_URL}?{urlencode(params)}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "bitirme-projesi/0.1",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise GetXAPIRequestError(
                f"GetXAPI request failed: {exc.code} {message}",
                FetchResult(tweets=tweets, api_calls=api_calls),
            ) from exc
        except (URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
            raise GetXAPIRequestError(
                f"Could not connect to GetXAPI: {exc}",
                FetchResult(tweets=tweets, api_calls=api_calls),
            ) from exc

        api_calls += 1
        if debug:
            print(
                "GetXAPI page "
                f"{api_calls}: tweet_count={payload.get('tweet_count')}, "
                f"tweets={len(payload.get('tweets', []))}, "
                f"has_more={payload.get('has_more')}, "
                f"next_cursor={bool(payload.get('next_cursor'))}"
            )

        for tweet in payload.get("tweets", []):
            tweet_id = str(tweet.get("id", ""))
            if tweet_id and tweet_id not in seen_ids:
                seen_ids.add(tweet_id)
                tweets.append(tweet)
                if len(tweets) >= max_tweets:
                    break

        cursor = payload.get("next_cursor")
        if not payload.get("has_more") or not cursor:
            break

    return FetchResult(tweets=tweets, api_calls=api_calls)


def fetch_tweets_for_date_range(
    query: str,
    start_date: date,
    end_date: date,
    api_key: str,
    product: str,
    tweets_per_day: int,
    max_api_calls: int | None = None,
    debug: bool = False,
) -> FetchResult:
    all_tweets: list[dict[str, Any]] = []
    total_calls = 0

    for day in iter_dates(start_date, end_date):
        if max_api_calls is not None and total_calls >= max_api_calls:
            print(f"Stopped before {day.isoformat()}: max API calls reached")
            break

        remaining_calls = None
        if max_api_calls is not None:
            remaining_calls = max_api_calls - total_calls

        daily_result = fetch_tweets_for_query(
            query=build_daily_query(query, day),
            api_key=api_key,
            product=product,
            max_tweets=tweets_per_day,
            max_api_calls=remaining_calls,
            debug=debug,
        )
        total_calls += daily_result.api_calls
        all_tweets.extend(daily_result.tweets)
        print(
            f"{day.isoformat()}: fetched {len(daily_result.tweets)} tweets "
            f"with {daily_result.api_calls} calls"
        )

    return FetchResult(tweets=all_tweets, api_calls=total_calls)


def download_tweets_for_date_range(
    query: str,
    start_date: date,
    end_date: date,
    api_key: str,
    product: str,
    tweets_per_day: int,
    output_path: Path,
    max_api_calls: int | None = None,
    debug: bool = False,
    chunk_hours: int = DEFAULT_CHUNK_HOURS,
    max_calls_per_window: int = DEFAULT_MAX_CALLS_PER_WINDOW,
    sample_window_minutes: int | None = DEFAULT_SAMPLE_WINDOW_MINUTES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
    progress_path: Path | None = None,
    resume: bool = True,
    max_windows: int | None = DEFAULT_MAX_WINDOWS_PER_RUN,
    max_days: int | None = DEFAULT_MAX_DAYS_PER_RUN,
) -> DownloadSummary:
    total_tweets = 0
    total_calls = 0
    current_day: date | None = None
    current_day_tweets = 0
    current_day_windows = 0
    processed_windows = 0
    per_window_limit = tweets_per_window(tweets_per_day, chunk_hours)
    progress_path = progress_path or default_progress_path(output_path)

    if overwrite or not output_path.exists():
        save_raw_data(build_tweet_dataframe([], query, product), output_path)
    if overwrite or not progress_path.exists():
        save_progress_data(pd.DataFrame(columns=PROGRESS_COLUMNS), progress_path)

    completed_windows = set()
    if resume and not overwrite:
        completed_windows = load_completed_windows(progress_path, query, product)

    windows = iter_time_windows(start_date, end_date, chunk_hours)
    target_days = select_incomplete_days(
        windows,
        completed_windows,
        query,
        product,
        sample_window_minutes,
        random_seed,
        max_days,
    )

    for day, window_start, window_end in windows:
        if target_days is not None and day not in target_days:
            continue

        if current_day != day:
            if current_day is not None and current_day_windows:
                print(f"{current_day.isoformat()}: fetched {current_day_tweets} tweets")
            current_day = day
            current_day_tweets = 0
            current_day_windows = 0

        if max_api_calls is not None and total_calls >= max_api_calls:
            message = f"max API calls reached before {window_start.isoformat()}"
            print(f"Stopped: {message}")
            return DownloadSummary(total_tweets, total_calls, message)

        if current_day_tweets >= tweets_per_day:
            continue

        remaining_calls = None
        if max_api_calls is not None:
            remaining_calls = max_api_calls - total_calls
        if remaining_calls is None:
            remaining_calls = max_calls_per_window
        else:
            remaining_calls = min(remaining_calls, max_calls_per_window)

        remaining_day_tweets = tweets_per_day - current_day_tweets
        sample_start, sample_end = choose_sample_window(
            window_start,
            window_end,
            sample_window_minutes,
            random_seed,
        )
        completed_key = window_key(query, product, sample_start, sample_end)
        if completed_key in completed_windows:
            if debug:
                print(f"Skipping completed window {sample_start.isoformat()} to {sample_end.isoformat()}")
            continue

        try:
            daily_result = fetch_tweets_for_query(
                query=build_time_window_query(query, sample_start, sample_end),
                api_key=api_key,
                product=product,
                max_tweets=min(per_window_limit, remaining_day_tweets),
                max_api_calls=remaining_calls,
                debug=debug,
            )
        except GetXAPIRequestError as exc:
            total_calls += exc.partial_result.api_calls
            partial_data = build_tweet_dataframe(exc.partial_result.tweets, query, product)
            append_raw_data(partial_data, output_path)
            total_tweets += len(partial_data)
            current_day_tweets += len(partial_data)
            message = f"{sample_start.isoformat()} to {sample_end.isoformat()}: {exc}"
            append_progress_row(
                progress_path,
                query,
                product,
                sample_start,
                sample_end,
                len(partial_data),
                exc.partial_result.api_calls,
                "failed",
                str(exc),
            )
            print(f"Stopped: {message}")
            return DownloadSummary(total_tweets, total_calls, message)

        total_calls += daily_result.api_calls
        daily_data = build_tweet_dataframe(daily_result.tweets, query, product)
        append_raw_data(daily_data, output_path)
        total_tweets += len(daily_data)
        current_day_tweets += len(daily_data)
        append_progress_row(
            progress_path,
            query,
            product,
            sample_start,
            sample_end,
            len(daily_data),
            daily_result.api_calls,
            "completed",
            "",
        )
        processed_windows += 1
        current_day_windows += 1
        if debug:
            print(
                f"{sample_start.isoformat()} to {sample_end.isoformat()}: "
                f"fetched {len(daily_data)} tweets with {daily_result.api_calls} calls"
            )

        if max_windows is not None and processed_windows >= max_windows:
            if current_day is not None:
                print(f"{current_day.isoformat()}: fetched {current_day_tweets} tweets")
            message = f"max windows reached after {sample_start.isoformat()} to {sample_end.isoformat()}"
            print(f"Stopped: {message}")
            return DownloadSummary(total_tweets, total_calls, message)

    if current_day is not None and current_day_windows:
        print(f"{current_day.isoformat()}: fetched {current_day_tweets} tweets")
    return DownloadSummary(total_tweets, total_calls)


def build_tweet_dataframe(tweets: list[dict[str, Any]], query: str, product: str) -> pd.DataFrame:
    columns = [
        "tweet_id",
        "url",
        "created_at",
        "date",
        "author_id",
        "author_username",
        "author_name",
        "author_followers",
        "author_verified",
        "lang",
        "text",
        "retweet_count",
        "reply_count",
        "like_count",
        "quote_count",
        "bookmark_count",
        "view_count",
        "is_reply",
        "conversation_id",
        "source",
        "media",
        "query",
        "product",
    ]

    rows: list[dict[str, Any]] = []
    for tweet in tweets:
        author = tweet.get("author") or {}
        rows.append(
            {
                "tweet_id": tweet.get("id"),
                "url": tweet.get("url") or tweet.get("twitterUrl"),
                "created_at": _parse_twitter_datetime(tweet.get("createdAt")),
                "date": _date_from_created_at(tweet.get("createdAt")),
                "author_id": author.get("id"),
                "author_username": author.get("userName"),
                "author_name": author.get("name"),
                "author_followers": author.get("followers"),
                "author_verified": author.get("isVerified")
                or author.get("isBlueVerified"),
                "lang": tweet.get("lang"),
                "text": normalize_tweet_text(tweet.get("text")),
                "retweet_count": tweet.get("retweetCount", 0),
                "reply_count": tweet.get("replyCount", 0),
                "like_count": tweet.get("likeCount", 0),
                "quote_count": tweet.get("quoteCount", 0),
                "bookmark_count": tweet.get("bookmarkCount", 0),
                "view_count": tweet.get("viewCount", 0),
                "is_reply": tweet.get("isReply"),
                "conversation_id": tweet.get("conversationId"),
                "source": tweet.get("source"),
                "media": json.dumps(tweet.get("media", []), ensure_ascii=True),
                "query": query,
                "product": product,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def save_raw_data(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def default_progress_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_progress.csv")


def window_key(
    query: str,
    product: str,
    window_start: datetime,
    window_end: datetime,
) -> tuple[str, str, str, str]:
    return (
        query,
        product,
        window_start.isoformat(),
        window_end.isoformat(),
    )


def save_progress_data(data: pd.DataFrame, progress_path: Path) -> Path:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(progress_path, index=False)
    return progress_path


def append_progress_row(
    progress_path: Path,
    query: str,
    product: str,
    window_start: datetime,
    window_end: datetime,
    tweet_count: int,
    api_calls: int,
    status: str,
    message: str,
) -> Path:
    row = pd.DataFrame(
        [
            {
                "query": query,
                "product": product,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "tweet_count": tweet_count,
                "api_calls": api_calls,
                "status": status,
                "message": message,
            }
        ],
        columns=PROGRESS_COLUMNS,
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not progress_path.exists() or progress_path.stat().st_size == 0
    row.to_csv(progress_path, mode="a", header=should_write_header, index=False)
    return progress_path


def load_completed_windows(
    progress_path: Path,
    query: str,
    product: str,
) -> set[tuple[str, str, str, str]]:
    if not progress_path.exists():
        return set()

    progress = pd.read_csv(progress_path)
    if progress.empty:
        return set()

    progress = progress[
        (progress["query"] == query)
        & (progress["product"] == product)
        & (progress["status"] == "completed")
    ]
    return {
        (row.query, row.product, row.window_start, row.window_end)
        for row in progress.itertuples(index=False)
    }


def append_raw_data(data: pd.DataFrame, output_path: Path) -> Path:
    if data.empty:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, mode="a", header=False, index=False)
    return output_path


def estimate_cost(api_calls: int) -> float:
    return api_calls * GETXAPI_COST_PER_CALL_USD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download filtered Bitcoin tweets from GetXAPI."
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
        default=None,
        help="Base GetXAPI advanced-search query before daily since/until filters.",
    )
    parser.add_argument(
        "--product",
        choices=["Top", "Latest"],
        default=DEFAULT_PRODUCT,
        help=f"GetXAPI search product. Default: {DEFAULT_PRODUCT}.",
    )
    parser.add_argument(
        "--no-sentiment-terms",
        action="store_true",
        help="Deprecated: sentiment terms are already disabled by default.",
    )
    parser.add_argument(
        "--sentiment-terms",
        action="store_true",
        help="Require market sentiment words like bullish, bearish, pump, dump, buy, or sell.",
    )
    parser.add_argument(
        "--exclude-links",
        action="store_true",
        help="Exclude tweets containing links. Disabled by default because it can remove too many crypto tweets.",
    )
    parser.add_argument(
        "--min-faves",
        type=int,
        help="Optional minimum like count filter. Disabled by default because it can remove too many historical tweets.",
    )
    parser.add_argument(
        "--tweets-per-day",
        type=int,
        default=DEFAULT_TWEETS_PER_DAY,
        help=f"Maximum tweets to fetch per day. Default: {DEFAULT_TWEETS_PER_DAY}.",
    )
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=DEFAULT_CHUNK_HOURS,
        help=(
            "Hours per GetXAPI search window. Smaller values spread tweets across the day. "
            f"Default: {DEFAULT_CHUNK_HOURS}."
        ),
    )
    parser.add_argument(
        "--max-calls-per-window",
        type=int,
        default=DEFAULT_MAX_CALLS_PER_WINDOW,
        help=(
            "Maximum GetXAPI calls per time window. "
            f"Default: {DEFAULT_MAX_CALLS_PER_WINDOW}."
        ),
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_RUN,
        help=(
            "Maximum uncompleted time windows to process per run. "
            f"Default: {DEFAULT_MAX_WINDOWS_PER_RUN}. Use 0 for no window limit."
        ),
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=DEFAULT_MAX_DAYS_PER_RUN,
        help=(
            "Maximum incomplete days to process per run. "
            f"Default: {DEFAULT_MAX_DAYS_PER_RUN}. Use 0 to process all remaining days."
        ),
    )
    parser.add_argument(
        "--sample-window-minutes",
        type=int,
        default=DEFAULT_SAMPLE_WINDOW_MINUTES,
        help=(
            "Random sub-window size inside each chunk. Use 0 to search the full chunk. "
            f"Default: {DEFAULT_SAMPLE_WINDOW_MINUTES}."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Seed used for deterministic random sub-windows. Default: {DEFAULT_RANDOM_SEED}.",
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        help="Optional hard cap on GetXAPI calls for budget control.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print GetXAPI pagination fields for each response page.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        help="Progress CSV path used for resumable runs. Defaults to <output>_progress.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output CSV before downloading. By default, new rows are appended.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip windows already marked completed in the progress CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_start_date, price_end_date = read_price_date_range(args.price_input)
    start_date = args.start_date or price_start_date
    end_date = args.end_date or price_end_date
    query = args.query or build_query(
        include_sentiment_terms=args.sentiment_terms and not args.no_sentiment_terms,
        exclude_links=args.exclude_links,
        min_faves=args.min_faves,
    )

    summary = download_tweets_for_date_range(
        query=query,
        start_date=start_date,
        end_date=end_date,
        api_key=get_api_key(),
        product=args.product,
        tweets_per_day=args.tweets_per_day,
        output_path=args.output,
        max_api_calls=args.max_api_calls,
        debug=args.debug,
        chunk_hours=args.chunk_hours,
        max_calls_per_window=args.max_calls_per_window,
        sample_window_minutes=args.sample_window_minutes or None,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
        progress_path=args.progress_output,
        resume=not args.no_resume,
        max_windows=None if args.max_windows == 0 else args.max_windows,
        max_days=None if args.max_days == 0 else args.max_days,
    )
    print(
        f"Saved {summary.tweet_count} raw tweets to {args.output}. "
        f"API calls: {summary.api_calls}. Estimated GetXAPI cost: ${estimate_cost(summary.api_calls):.2f}."
    )
    if summary.stopped_reason:
        print(f"Stopped early: {summary.stopped_reason}")


if __name__ == "__main__":
    main()
