"""Visualization helpers: training history, predictions, and ablation comparison charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no popup windows in background tasks
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history,
    symbol: str = "",
    task: str = "classification",
    save_path: str | Path | None = None,
) -> None:
    """Plot training vs. validation loss and accuracy/MAE curves."""
    metric = "accuracy" if task == "classification" else "mae"
    metric_label = "Accuracy" if task == "classification" else "MAE"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title(f"{symbol} — Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if metric in history.history:
        ax2.plot(history.history[metric], label=f"Train {metric_label}")
        ax2.plot(history.history[f"val_{metric}"], label=f"Val {metric_label}")
        ax2.set_title(f"{symbol} — {metric_label}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(metric_label)
        if task == "classification":
            ax2.axhline(0.65, color="red", linestyle="--", alpha=0.6, label="Target 65%")
        ax2.legend()
        ax2.grid(alpha=0.3)

    plt.suptitle(symbol, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_predictions(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    symbol: str = "",
    dates=None,
    save_path: str | Path | None = None,
) -> None:
    """Plot actual vs. predicted direction and the raw predicted probability."""
    y_pred = (np.array(y_pred_prob).flatten() >= 0.5).astype(int)
    y_true = np.array(y_true).flatten()
    x = np.arange(len(y_true)) if dates is None else dates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(x, y_true, label="Actual", alpha=0.8, linewidth=1.2)
    ax1.plot(x, y_pred, label="Predicted", alpha=0.6, linestyle="--", linewidth=1.2)
    ax1.set_title(f"{symbol} — Direction: Actual vs Predicted")
    ax1.set_ylabel("1=Up, 0=Down")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(x, np.array(y_pred_prob).flatten(), color="purple", label="P(Up)", linewidth=1.0)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Predicted Probability")
    ax2.set_xlabel("Bar")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_ablation_comparison(
    results: dict,
    symbol: str = "",
    task: str = "classification",
    target_col: str = "Next_Close",
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing the three ablation scenarios side by side."""
    names = list(results.keys())
    short_names = ["Price\nOnly", "Price +\nTechnical", "Price + Tech\n+ Sentiment"]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    if task == "classification":
        metric_a = [results[n].get("accuracy", 0) * 100 for n in names]
        metric_b = [results[n].get("f1", 0) * 100 for n in names]
        label_a, label_b = "Direction Accuracy %", "F1 %"
    elif target_col == "Target_Return":
        # For return prediction: direction accuracy is the key metric
        metric_a = [results[n].get("direction_accuracy", 0) * 100 for n in names]
        metric_b = [results[n].get("mae", 0) * 100 for n in names]
        label_a, label_b = "Direction Accuracy %", "MAE × 100"
    else:
        # For price prediction: MAPE and direction accuracy
        metric_a = [results[n].get("mape", 0) for n in names]
        metric_b = [results[n].get("direction_accuracy", 0) * 100 for n in names]
        label_a, label_b = "MAPE %", "Direction Accuracy %"

    x = np.arange(len(names))
    width = 0.35
    bars_a = ax.bar(x - width / 2, metric_a, width, label=label_a, color="#4C72B0")
    bars_b = ax.bar(x + width / 2, metric_b, width, label=label_b, color="#DD8452")

    if task == "classification":
        ax.axhline(50, color="gray", linestyle="--", alpha=0.6, label="Random baseline 50%")
        ax.axhline(65, color="red", linestyle="--", alpha=0.7, label="Thesis target 65%")
    elif target_col == "Target_Return":
        ax.axhline(50, color="gray", linestyle="--", alpha=0.6, label="Random baseline 50%")

    ax.set_title(f"{symbol} — Ablation: Feature Set Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names[:len(names)])
    ax.set_ylim(0, 100 if task == "classification" or target_col == "Target_Return" else None)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars_a, fmt="%.1f", padding=3)
    ax.bar_label(bars_b, fmt="%.1f", padding=3)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save_or_show(save_path)


def plot_ablation_histories(
    histories: dict,
    symbol: str = "",
    task: str = "regression",
    save_path: str | Path | None = None,
) -> None:
    """Plot training & validation loss curves for all 3 ablation scenarios on one figure."""
    scenario_labels = {
        "price_only":                "Price Only",
        "price_technical":           "Price + Technical",
        "price_technical_sentiment": "Price + Tech + Sentiment",
    }
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    metric = "mae" if task == "regression" else "accuracy"
    metric_label = "MAE" if task == "regression" else "Accuracy"

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4))

    for idx, (name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        label = scenario_labels.get(name, name)
        h = history.history

        axes[0].plot(h["loss"],     color=color, linewidth=1.5, label=f"{label} — train")
        axes[0].plot(h["val_loss"], color=color, linewidth=1.0, linestyle="--", alpha=0.6, label=f"{label} — val")

        if metric in h:
            axes[1].plot(h[metric],          color=color, linewidth=1.5)
            axes[1].plot(h[f"val_{metric}"], color=color, linewidth=1.0, linestyle="--", alpha=0.6)

    axes[0].set_title("Loss (MSE)" if task == "regression" else "Loss (BCE)", fontsize=11)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].set_title(metric_label, fontsize=11)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel(metric_label)
    axes[1].grid(alpha=0.3)

    # Single shared legend below both subplots — 3 columns (one per scenario)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(f"{symbol} — Training History by Scenario", fontsize=13, fontweight="bold")
    # rect=[left, bottom, right, top] — reserve space at bottom for legend, top for suptitle
    plt.tight_layout(rect=[0, 0.18, 1, 0.95])
    _save_or_show(save_path)


def plot_price_vs_prediction(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    symbol: str = "",
    dates=None,
    save_path: str | Path | None = None,
) -> None:
    """Plot actual vs. predicted close prices (for regression task)."""
    x = np.arange(len(actual_prices)) if dates is None else dates

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, actual_prices, label="Actual", linewidth=1.2)
    ax.plot(x, predicted_prices, label="Predicted", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_title(f"{symbol} — Actual vs Predicted Close Price")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(save_path: str | Path | None) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        print(f"Chart saved -> {path}")
    plt.show()
    plt.close()
