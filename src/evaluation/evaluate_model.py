"""Run 3-scenario ablation study and evaluate a trained model.

Scenarios (from thesis Section 3.7):
  1. price_only          — close + volume
  2. price_technical     — + all technical indicators
  3. price_technical_sentiment — + VADER sentiment score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.metrics import (
    classification_metrics,
    print_classification_report,
    regression_metrics,
    sentiment_price_correlation,
)
from src.features.technical_indicators import (
    FEATURES_PRICE_ONLY,
    FEATURES_TECHNICAL,
    FEATURES_WITH_SENTIMENT,
)
from src.models.lstm_model import build_lstm, prepare_data
from src.models.train_model import train
from src.visualization.plot_results import plot_ablation_histories


# Feature sets for the three ablation scenarios
SCENARIOS: dict[str, list[str]] = {
    "price_only":                FEATURES_PRICE_ONLY,
    "price_technical":           FEATURES_TECHNICAL,
    "price_technical_sentiment": FEATURES_WITH_SENTIMENT,
}


def run_ablation(
    df: pd.DataFrame,
    look_back: int = 20,
    units: list[int] | None = None,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    reduce_lr_patience: int = 7,
    task: str = "classification",
    symbol: str = "",
    verbose: int = 0,
    history_save_path=None,
    target_col: str | None = None,
) -> dict[str, dict]:
    """Train and evaluate the model under all three feature-set scenarios.

    Returns a dict mapping scenario name -> metrics dict.
    Prints a summary table when done.
    """
    if units is None:
        units = [128, 64]

    if target_col is None:
        target_col = "Target_Class" if task == "classification" else "Next_Close"
    # Returns (Target_Return) are already small (-0.3 to +0.3); only scale absolute prices
    scale_target = (task == "regression") and (target_col == "Next_Close")
    results: dict[str, dict] = {}
    histories: dict[str, object] = {}

    for name, feature_list in SCENARIOS.items():
        available = [f for f in feature_list if f in df.columns]
        print(f"\n--- [{symbol}] Scenario: {name} ({len(available)} features) ---")

        X_tr, y_tr, X_val, y_val, X_te, y_te, _, target_scaler = prepare_data(
            df, available, target_col, look_back, scale_target=scale_target
        )
        model = build_lstm(
            input_shape=(look_back, len(available)),
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            task=task,
        )

        # Balanced class weights prevent the model from collapsing to the majority class
        if task == "classification":
            classes = np.unique(y_tr).astype(int)
            weights = compute_class_weight("balanced", classes=classes, y=y_tr.astype(int))
            cw = dict(zip(classes, weights))
        else:
            cw = None

        hist = train(
            model, X_tr, y_tr, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            reduce_lr_patience=reduce_lr_patience,
            verbose=verbose,
            class_weight=cw,
        )
        histories[name] = hist

        y_pred = model.predict(X_te, verbose=0).flatten()

        if task == "classification":
            m = classification_metrics(y_te, y_pred)
            print_classification_report(m, symbol=symbol, scenario=name)
        else:
            # Inverse-transform scaled predictions back to original price scale
            if target_scaler is not None:
                y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_te   = target_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
            m = regression_metrics(y_te, y_pred)

        m["features_used"] = available
        m["n_features"] = len(available)
        results[name] = m

    _print_ablation_summary(results, symbol, task, target_col)

    if history_save_path is not None:
        plot_ablation_histories(histories, symbol=symbol, task=task,
                                save_path=history_save_path)

    return results


def evaluate_single(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "classification",
    symbol: str = "",
) -> dict:
    """Evaluate a pre-trained model on a test set."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    if task == "classification":
        m = classification_metrics(y_test, y_pred)
        print_classification_report(m, symbol=symbol)
    else:
        m = regression_metrics(y_test, y_pred)
    return m


def run_sentiment_correlation(
    df: pd.DataFrame,
    symbol: str = "",
) -> dict:
    """Compute Pearson correlation between Sentiment and Target_Return columns."""
    if "Sentiment" not in df.columns or "Target_Return" not in df.columns:
        print("[sentiment_correlation] Missing Sentiment or Target_Return columns.")
        return {"pearson_r": float("nan"), "p_value": float("nan")}

    result = sentiment_price_correlation(
        df["Sentiment"].values,
        df["Target_Return"].values,
    )
    print(
        f"[{symbol}] Sentiment–Return correlation: "
        f"r={result['pearson_r']:.4f}  p={result['p_value']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_ablation_summary(results: dict, symbol: str, task: str, target_col: str = "Next_Close") -> None:
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY  |  {symbol}  |  task={task}")
    print(f"{'='*60}")
    print(f"  {'Scenario':<35} {'Accuracy':>9} {'F1':>8}")
    print(f"  {'-'*55}")
    for name, m in results.items():
        if task == "classification":
            acc = m.get("accuracy", float("nan"))
            f1  = m.get("f1", float("nan"))
            flag = " ✓" if acc >= 0.65 else ""
            print(f"  {name:<35} {acc*100:>8.1f}% {f1*100:>7.1f}%{flag}")
        else:
            da  = m.get("direction_accuracy", float("nan"))
            r2  = m.get("r2", float("nan"))
            mae = m.get("mae", float("nan"))
            if target_col == "Target_Return":
                # MAPE is meaningless for near-zero returns; show MAE and direction
                print(f"  {name:<35} MAE={mae:>8.4f}  R²={r2:>6.4f}  Dir={da*100:>5.1f}%")
            else:
                mape = m.get("mape", float("nan"))
                print(f"  {name:<35} MAPE={mape:>6.2f}%  R²={r2:>6.4f}  Dir={da*100:>5.1f}%")
    print(f"{'='*60}\n")
