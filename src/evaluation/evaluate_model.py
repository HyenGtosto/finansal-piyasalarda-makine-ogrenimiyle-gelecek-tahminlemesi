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
from src.visualization.plot_results import plot_ablation_histories, plot_feature_importance


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
    importance_save_path=None,
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
        m["model"] = model
        m["X_te"]  = X_te
        m["y_te"]  = y_te
        results[name] = m

    _print_ablation_summary(results, symbol, task, target_col)

    if history_save_path is not None:
        plot_ablation_histories(histories, symbol=symbol, task=task,
                                save_path=history_save_path)

    if importance_save_path is not None and task == "regression":
        # Run permutation importance on the richest scenario (price_technical_sentiment)
        best_scenario = "price_technical_sentiment"
        if best_scenario in results and "model" in results[best_scenario]:
            imp_model    = results[best_scenario]["model"]
            imp_X        = results[best_scenario]["X_te"]
            imp_y        = results[best_scenario]["y_te"]
            imp_features = results[best_scenario]["features_used"]
            print(f"\n[{symbol}] Computing permutation feature importance...")
            importance = compute_permutation_importance(
                imp_model, imp_X, imp_y, imp_features, target_col=target_col
            )
            plot_feature_importance(
                importance, symbol=symbol, save_path=importance_save_path
            )

    return results


def compute_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    target_col: str = "Next_Close",
    n_repeats: int = 5,
) -> dict[str, float]:
    """Measure each feature's importance by shuffling it and recording accuracy drop.

    Shuffles one feature at a time across all test samples, runs prediction,
    computes direction accuracy, and compares to the baseline. Repeats n_repeats
    times per feature and averages to reduce randomness.

    Returns a dict mapping feature_name -> importance score (baseline_acc - shuffled_acc).
    Positive = feature helped; near zero = feature didn't matter.
    """
    y_pred_base = model.predict(X_test, verbose=0).flatten()
    y_ref = y_test[:-1]
    y_true_dir = np.sign(y_test[1:] - y_ref)
    pred_dir    = np.sign(y_pred_base[1:] - y_ref)
    baseline_acc = float(np.mean(y_true_dir == pred_dir))

    importance: dict[str, float] = {}
    rng = np.random.default_rng(42)

    for feat_idx, feat_name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            # Shuffle this feature's values across all samples (break its signal)
            perm = rng.permutation(X_shuffled.shape[0])
            X_shuffled[:, :, feat_idx] = X_shuffled[perm, :, feat_idx]

            y_pred_shuf = model.predict(X_shuffled, verbose=0).flatten()
            pred_dir_shuf = np.sign(y_pred_shuf[1:] - y_ref)
            shuffled_acc = float(np.mean(y_true_dir == pred_dir_shuf))
            drops.append(baseline_acc - shuffled_acc)

        importance[feat_name] = float(np.mean(drops))

    return importance


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
