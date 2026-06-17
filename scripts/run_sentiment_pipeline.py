"""Score all tweets with VADER and produce daily/4H sentiment aggregation files.

This script:
  1. Loads data/interim/bitcoin_tweets_cleaned.csv (tweet text)
  2. Scores every tweet with VADER compound score
  3. Overwrites data/interim/bitcoin_tweets_sentiment.csv with real scores
  4. Produces data/processed/bitcoin_daily_sentiment.csv  (daily bars)
     OR  data/processed/bitcoin_4h_sentiment.csv          (4-hour bars)

Usage:
  python scripts/run_sentiment_pipeline.py
  python scripts/run_sentiment_pipeline.py --symbol bitcoin
  python scripts/run_sentiment_pipeline.py --interval 4h
  python scripts/run_sentiment_pipeline.py --no-weighted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sentiment_features import (
    aggregate_4h_sentiment,
    aggregate_daily_sentiment,
    load_and_score_tweets,
    save_4h_sentiment,
    save_daily_sentiment,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VADER sentiment pipeline on tweet data.")
    p.add_argument("--symbol", type=str, default="bitcoin",
                   help="Asset name prefix for the CSV files (default: bitcoin)")
    p.add_argument("--interval", type=str, default="1d", choices=["1d", "4h"],
                   help="Aggregation granularity: 1d (daily) or 4h (4-hour). Default: 1d")
    p.add_argument("--no-weighted", action="store_true",
                   help="Disable engagement-weighted sentiment; use simple mean instead")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol
    interval = args.interval
    use_weighted = not args.no_weighted

    interim_dir   = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    cleaned_csv   = interim_dir / f"{symbol}_tweets_cleaned.csv"
    sentiment_csv = interim_dir / f"{symbol}_tweets_sentiment.csv"

    # ------------------------------------------------------------------
    # 1. Score tweets
    # ------------------------------------------------------------------
    print(f"\n[1/3] Loading and scoring tweets from {cleaned_csv.name} ...")
    if not cleaned_csv.exists():
        print(f"ERROR: {cleaned_csv} not found. Run the tweet cleaning pipeline first.")
        sys.exit(1)

    scored_df = load_and_score_tweets(
        cleaned_csv=cleaned_csv,
        sentiment_csv=sentiment_csv,
        text_col="text",
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 2. Overwrite sentiment CSV with real scores
    # ------------------------------------------------------------------
    print(f"\n[2/3] Saving scored tweets -> {sentiment_csv.name}")

    save_cols = ["tweet_id", "created_at", "sentiment_score", "sentiment_label"]
    if "engagement_weight" in scored_df.columns:
        save_cols.append("engagement_weight")

    scored_df[save_cols].to_csv(sentiment_csv, index=False)

    # Quick sanity check
    nonzero = (scored_df["sentiment_score"] != 0).sum()
    pos = (scored_df["sentiment_label"] == "positive").sum()
    neg = (scored_df["sentiment_label"] == "negative").sum()
    neu = (scored_df["sentiment_label"] == "neutral").sum()
    print(f"  Non-zero scores : {nonzero:,} / {len(scored_df):,} ({nonzero/len(scored_df)*100:.1f}%)")
    print(f"  Positive        : {pos:,} ({pos/len(scored_df)*100:.1f}%)")
    print(f"  Negative        : {neg:,} ({neg/len(scored_df)*100:.1f}%)")
    print(f"  Neutral         : {neu:,} ({neu/len(scored_df)*100:.1f}%)")
    print(f"  Mean compound   : {scored_df['sentiment_score'].mean():.4f}")

    # ------------------------------------------------------------------
    # 3. Aggregate to requested granularity
    # ------------------------------------------------------------------
    weight_col = "engagement_weight" if (use_weighted and "engagement_weight" in scored_df.columns) else None

    if interval == "4h":
        print(f"\n[3/3] Aggregating to 4-hour sentiment buckets ...")
        agg = aggregate_4h_sentiment(
            scored_df,
            date_col="created_at",
            score_col="sentiment_score",
            weight_col=weight_col,
        )
        out_path = processed_dir / f"{symbol}_4h_sentiment.csv"
        save_4h_sentiment(agg, out_path)

        print(f"\nBucket range  : {agg.index.min()} -> {agg.index.max()}")
        print(f"4H buckets    : {len(agg)}")
        print(f"Mean sentiment: {agg['sentiment_mean'].mean():.4f}")
        if "weighted_sentiment_mean" in agg.columns:
            print(f"Mean weighted : {agg['weighted_sentiment_mean'].mean():.4f}")

        print("\nDone. You can now run:")
        print(f"  python scripts/run_training.py --symbol BTC-USD --ablation --task regression --interval 4h")

    else:
        print(f"\n[3/3] Aggregating to daily sentiment ...")
        daily = aggregate_daily_sentiment(
            scored_df,
            date_col="created_at",
            score_col="sentiment_score",
            weight_col=weight_col,
        )
        out_path = processed_dir / f"{symbol}_daily_sentiment.csv"
        save_daily_sentiment(daily, out_path)

        print(f"\nDate range  : {daily.index.min().date()} -> {daily.index.max().date()}")
        print(f"Days with data: {len(daily)}")
        print(f"Mean daily sentiment : {daily['sentiment_mean'].mean():.4f}")
        if "weighted_sentiment_mean" in daily.columns:
            print(f"Mean weighted sent.  : {daily['weighted_sentiment_mean'].mean():.4f}")

        print("\nDone. You can now run:")
        print(f"  python scripts/run_training.py --symbol BTC-USD --ablation")


if __name__ == "__main__":
    main()
