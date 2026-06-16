"""Plot model comparison metrics and confusion matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_PATH = Path("reports/model_results.csv")
DEFAULT_CONFUSION_PATH = Path("reports/confusion_matrices.json")
DEFAULT_OUTPUT_DIR = Path("reports/figures")
METRIC_COLUMNS = ["accuracy", "roc_auc", "f1", "precision", "recall"]


def plot_metric_comparison(results: pd.DataFrame, output_dir: Path) -> Path:
    trained = results[results["status"] == "trained"].copy()
    melted = trained.melt(
        id_vars=["feature_set", "model"],
        value_vars=METRIC_COLUMNS,
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(14, 7))
    sns.barplot(data=melted, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Model Metrics Across All Feature Sets")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_path = output_dir / "model_metrics_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_feature_set_auc(results: pd.DataFrame, output_dir: Path) -> Path:
    trained = results[results["status"] == "trained"].copy()

    plt.figure(figsize=(13, 6))
    sns.barplot(data=trained, x="feature_set", y="roc_auc", hue="model")
    plt.ylim(0, 1)
    plt.title("ROC-AUC by Feature Set")
    plt.xlabel("Feature Set")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_path = output_dir / "roc_auc_by_feature_set.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_model_heatmap(results: pd.DataFrame, output_dir: Path) -> Path:
    trained = results[results["status"] == "trained"].copy()
    heatmap_data = trained.pivot_table(
        index=["feature_set", "model"],
        values=METRIC_COLUMNS,
        aggfunc="first",
    )

    plt.figure(figsize=(10, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1)
    plt.title("Model Metric Heatmap")
    plt.xlabel("Metric")
    plt.ylabel("Feature Set / Model")
    plt.tight_layout()

    output_path = output_dir / "model_metric_heatmap.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_confusion_matrices(confusion_path: Path, output_dir: Path) -> list[Path]:
    if not confusion_path.exists():
        return []

    confusion_data = json.loads(confusion_path.read_text(encoding="utf-8"))
    output_paths: list[Path] = []
    for name, matrix in confusion_data.items():
        safe_name = (
            name.replace(":", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        plt.figure(figsize=(4, 3.5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        plt.title(name)
        plt.tight_layout()

        output_path = output_dir / f"confusion_matrix_{safe_name}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        output_paths.append(output_path)

    return output_paths


def create_model_result_plots(
    results_path: Path,
    confusion_path: Path,
    output_dir: Path,
) -> list[Path]:
    results = pd.read_csv(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        plot_metric_comparison(results, output_dir),
        plot_feature_set_auc(results, output_dir),
        plot_model_heatmap(results, output_dir),
    ]
    output_paths.extend(plot_confusion_matrices(confusion_path, output_dir))
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model training results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Model metrics CSV path. Default: {DEFAULT_RESULTS_PATH}.",
    )
    parser.add_argument(
        "--confusion",
        type=Path,
        default=DEFAULT_CONFUSION_PATH,
        help=f"Confusion matrices JSON path. Default: {DEFAULT_CONFUSION_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Figure output directory. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = create_model_result_plots(args.results, args.confusion, args.output_dir)
    print("Saved figures:")
    for output_path in output_paths:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()
