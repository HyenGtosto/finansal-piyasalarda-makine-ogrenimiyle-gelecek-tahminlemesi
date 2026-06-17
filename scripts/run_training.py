"""Main training entry point.

Usage examples:
  # Train classification on BTC, single run with full feature set
  python scripts/run_training.py --symbol BTC-USD

  # Run 3-scenario ablation for NVDA
  python scripts/run_training.py --symbol NVDA --ablation

  # Regression (next close price) for Ethereum
  python scripts/run_training.py --symbol ETH-USD --task regression

  # Train all assets defined in config
  python scripts/run_training.py --all

  # Force re-download of price data
  python scripts/run_training.py --symbol AAPL --force-download
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Fix random seeds for reproducible results across runs
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Make project root importable regardless of where the script is called from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.download_prices import get_prices
from src.data.load_local_prices import has_local_prices, get_local_prices
from src.evaluation.evaluate_model import (
    evaluate_single,
    run_ablation,
    run_sentiment_correlation,
)
from src.features.sentiment_features import (
    load_daily_sentiment_series,
    load_4h_sentiment_series,
    load_nvda_weekly_sentiment_series,
)
from src.features.technical_indicators import (
    FEATURES_WITH_SENTIMENT,
    build_features,
)
from src.models.lstm_model import build_lstm, prepare_data
from src.models.train_model import train
from src.visualization.plot_results import (
    plot_ablation_comparison,
    plot_predictions,
    plot_training_history,
)

# Maps (symbol, interval) -> sentiment CSV produced by run_sentiment_pipeline.py
# Falls back to the plain string form for daily (backward compat).
SENTIMENT_CSV_MAP: dict[str, str] = {
    "BTC-USD":        "data/processed/bitcoin_daily_sentiment.csv",
    "BTC-USD__4h":    "data/processed/bitcoin_4h_sentiment.csv",
    "ETH-USD":        "data/processed/ethereum_daily_sentiment.csv",
    "NVDA":           "data/processed/nvidia_weekly_sentiment.csv",
}

# Symbols that use a custom loader instead of load_daily_sentiment_series
SENTIMENT_LOADERS: dict[str, object] = {
    "NVDA": load_nvda_weekly_sentiment_series,
}

# look_back override for 4H bars: 30 bars = 5 calendar days (6 bars/day)
LOOK_BACK_4H = 30


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(PROJECT_ROOT / path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-symbol pipeline
# ---------------------------------------------------------------------------

def run_symbol(
    symbol: str,
    cfg: dict,
    task: str = "classification",
    ablation: bool = False,
    force_download: bool = False,
    interval_override: str | None = None,
) -> None:
    data_cfg  = cfg["data"]
    # Merge per-asset overrides on top of the global model defaults
    asset_ov  = cfg.get("asset_overrides", {}).get(symbol, {})
    model_cfg = {**cfg["model"], **asset_ov}
    paths     = cfg["paths"]
    ind_cfg   = cfg.get("indicators", {})

    # Interval: CLI > asset_override > global config default
    interval = interval_override or asset_ov.get("interval") or data_cfg.get("interval", "1d")

    print(f"\n{'='*60}")
    print(f"  Symbol: {symbol}   Task: {task}   Ablation: {ablation}   Interval: {interval}")
    print(f"{'='*60}")

    # ---- 1. Load price data (local raw CSV if available, else yfinance) ----
    raw_dir = PROJECT_ROOT / paths["raw_data"]
    local_intervals = {"1d", "4h"}
    if not force_download and interval in local_intervals and has_local_prices(symbol, PROJECT_ROOT):
        print(f"[{symbol}] Using local raw price CSV ({interval} bars).")
        df = get_local_prices(symbol, PROJECT_ROOT, interval=interval)
    else:
        df = get_prices(
            symbol,
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
            data_dir=raw_dir,
            interval=interval,
            force_download=force_download,
        )

    # ---- 2. Load sentiment series if available for this symbol ----
    # For 4H bars, use the 4H-bucketed sentiment; fall back to daily.
    sentiment_series = None
    sentiment_key = f"{symbol}__{interval}" if interval != "1d" else symbol
    sentiment_csv_rel = SENTIMENT_CSV_MAP.get(sentiment_key) or SENTIMENT_CSV_MAP.get(symbol)
    if sentiment_csv_rel:
        sentiment_csv = PROJECT_ROOT / sentiment_csv_rel
        if sentiment_csv.exists():
            try:
                if interval == "4h":
                    sentiment_series = load_4h_sentiment_series(sentiment_csv)
                    print(f"Loaded 4H sentiment: {len(sentiment_series)} buckets  "
                          f"(mean={sentiment_series.mean():.4f})")
                elif symbol in SENTIMENT_LOADERS:
                    sentiment_series = SENTIMENT_LOADERS[symbol](sentiment_csv)
                    print(f"Loaded {symbol} sentiment: {len(sentiment_series)} periods  "
                          f"(mean={sentiment_series.mean():.4f})")
                else:
                    sentiment_series = load_daily_sentiment_series(sentiment_csv)
                    print(f"Loaded sentiment: {len(sentiment_series)} days  "
                          f"(mean={sentiment_series.mean():.4f})")
            except Exception as e:
                print(f"[warn] Could not load sentiment for {symbol}: {e}")
        else:
            hint = "--interval 4h" if interval == "4h" else ""
            print(f"[info] No sentiment file yet for {symbol}. "
                  f"Run: python scripts/run_sentiment_pipeline.py {hint}".strip())

    # ---- 2b. Clip price data to sentiment coverage window ----
    # When sentiment exists, trim price bars to the sentiment end date so the
    # test set never contains forward-filled (stale) sentiment values.
    if sentiment_series is not None:
        sent_end = sentiment_series.index.max()
        before   = len(df)
        df = df[df.index <= sent_end]
        if len(df) < before:
            print(f"[{symbol}] Price data clipped to sentiment end: "
                  f"{sent_end.date()}  ({before - len(df)} bars dropped)")

    # ---- 3. Feature engineering ----
    df = build_features(
        df,
        sentiment_series=sentiment_series,
        sma_windows=ind_cfg.get("sma_windows"),
        ema_windows=ind_cfg.get("ema_windows"),
        rsi_window=ind_cfg.get("rsi_window", 14),
    )
    print(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")

    reports_dir = PROJECT_ROOT / paths["reports"]
    safe_sym = symbol.replace("-", "_")

    # For 4H bars, use a longer look_back (30 bars = 5 calendar days of 6 bars each)
    effective_look_back = LOOK_BACK_4H if interval == "4h" else model_cfg["look_back"]

    # Target column: asset override ("return" -> Next_Return) takes precedence over task default
    asset_target = asset_ov.get("target")
    if task == "classification":
        target_col = "Target_Class"
    elif asset_target == "return":
        target_col = "Target_Return"
    else:
        target_col = "Next_Close"

    print(f"  Target column: {target_col}")

    # ---- 4a. Ablation (3 scenarios) ----
    if ablation:
        results = run_ablation(
            df,
            look_back=effective_look_back,
            units=model_cfg["lstm_units"],
            dropout=model_cfg["dropout"],
            learning_rate=model_cfg["learning_rate"],
            epochs=model_cfg["epochs"],
            batch_size=model_cfg["batch_size"],
            patience=model_cfg["early_stopping_patience"],
            reduce_lr_patience=model_cfg["reduce_lr_patience"],
            task=task,
            symbol=symbol,
            history_save_path=reports_dir / f"{safe_sym}_history.png",
            importance_save_path=reports_dir / f"{safe_sym}_importance.png",
            target_col=target_col,
        )
        plot_ablation_comparison(
            results,
            symbol=symbol,
            task=task,
            target_col=target_col,
            save_path=reports_dir / f"{safe_sym}_ablation.png",
        )
        # Sentiment–return correlation (only meaningful once VADER is wired in)
        run_sentiment_correlation(df, symbol=symbol)
        return

    # ---- 4b. Single run with full feature set ----
    features = [f for f in FEATURES_WITH_SENTIMENT if f in df.columns]

    X_tr, y_tr, X_val, y_val, X_te, y_te, scaler, target_scaler = prepare_data(
        df, features, target_col, effective_look_back,
        val_ratio=model_cfg["validation_split"],
        test_ratio=model_cfg["test_split"],
        scale_target=(task == "regression" and target_col == "Next_Close"),
    )

    model = build_lstm(
        input_shape=(effective_look_back, len(features)),
        units=model_cfg["lstm_units"],
        dropout=model_cfg["dropout"],
        learning_rate=model_cfg["learning_rate"],
        task=task,
    )
    model.summary()

    # Optionally save best weights
    models_dir = PROJECT_ROOT / paths.get("models", "data/models")
    model_save = models_dir / f"{safe_sym}_{task}_best.keras"

    history = train(
        model, X_tr, y_tr, X_val, y_val,
        epochs=model_cfg["epochs"],
        batch_size=model_cfg["batch_size"],
        patience=model_cfg["early_stopping_patience"],
        reduce_lr_patience=model_cfg["reduce_lr_patience"],
        model_save_path=model_save,
    )

    # ---- 5. Evaluate (inverse-transform regression targets back to original price scale) ----
    y_te_eval = y_te
    if task == "regression" and target_scaler is not None:
        y_pred_raw = model.predict(X_te, verbose=0).flatten()
        y_te_eval  = target_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
        y_pred_raw = target_scaler.inverse_transform(y_pred_raw.reshape(-1, 1)).flatten()
        from src.evaluation.metrics import regression_metrics, print_regression_report
        metrics = regression_metrics(y_te_eval, y_pred_raw)
        print_regression_report(metrics, symbol=symbol)
    else:
        metrics = evaluate_single(model, X_te, y_te_eval, task=task, symbol=symbol)

    # ---- 6. Plots ----
    plot_training_history(
        history,
        symbol=symbol,
        task=task,
        save_path=reports_dir / f"{safe_sym}_{task}_history.png",
    )

    if task == "classification":
        y_pred = model.predict(X_te, verbose=0).flatten()
        plot_predictions(
            y_te, y_pred,
            symbol=symbol,
            save_path=reports_dir / f"{safe_sym}_predictions.png",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM model for financial prediction.")
    p.add_argument("--symbol", type=str, default="BTC-USD",
                   help="Ticker symbol, e.g. BTC-USD, ETH-USD, NVDA, AAPL")
    p.add_argument("--task", type=str, default="classification",
                   choices=["classification", "regression"],
                   help="Prediction task: direction (classification) or price (regression)")
    p.add_argument("--ablation", action="store_true",
                   help="Run all 3 feature-set scenarios instead of a single full run")
    p.add_argument("--all", action="store_true",
                   help="Run for every asset defined in config/config.yaml")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download price data even if a cache file exists")
    p.add_argument("--interval", type=str, default=None,
                   help="Bar interval override: 1d (daily) or 1wk (weekly). "
                        "Overrides config and asset_overrides.")
    p.add_argument("--config", type=str, default="config/config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.all:
        symbols = [a["symbol"] for a in cfg.get("assets", [])]
        print(f"Running for all configured assets: {symbols}")
    else:
        symbols = [args.symbol]

    for sym in symbols:
        run_symbol(
            sym,
            cfg,
            task=args.task,
            ablation=args.ablation,
            force_download=args.force_download,
            interval_override=args.interval,
        )


if __name__ == "__main__":
    main()
