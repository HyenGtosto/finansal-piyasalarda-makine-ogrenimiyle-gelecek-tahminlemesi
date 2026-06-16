"""Train chronological Bitcoin direction models across feature stages."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


DEFAULT_DATASET_PATH = Path("data/processed/final_dataset.csv")
DEFAULT_RESULTS_PATH = Path("reports/model_results.csv")
DEFAULT_CONFUSION_PATH = Path("reports/confusion_matrices.json")
TARGET_COLUMN = "target_up_next_4h"
TIME_COLUMN = "window_start"
TARGET_COLUMNS = {"next_price_close", "next_4h_return", TARGET_COLUMN}
RANDOM_STATE = 42

BASELINE_MARKET_FEATURES = [
    "price_open",
    "price_high",
    "price_low",
    "price_close",
    "price_mean",
    "market_cap_mean",
    "volume_sum",
    "price_return",
    "price_range_pct",
    "price_return_mean_24h",
    "price_return_std_24h",
    "volume_sum_mean_24h",
    "volume_sum_zscore_24h",
]
SENTIMENT_CORE_FEATURES = [
    "tweet_count",
    "sentiment_mean",
    "sentiment_min",
    "sentiment_max",
    "sentiment_std",
    "positive_tweet_ratio",
    "negative_tweet_ratio",
    "engagement_weight_mean",
    "engagement_weight_sum",
    "weighted_sentiment_sum",
    "weighted_sentiment_mean",
]
SENTIMENT_DERIVED_FEATURES = [
    "weighted_sentiment_delta_4h",
    "sentiment_mean_delta_4h",
    "tweet_count_delta_4h",
    "engagement_weight_sum_delta_4h",
    "weighted_sentiment_mean_24h",
    "weighted_sentiment_std_24h",
    "tweet_count_mean_24h",
    "tweet_count_std_24h",
    "engagement_weight_sum_mean_24h",
    "engagement_weight_sum_std_24h",
    "weighted_sentiment_mean_7d",
    "tweet_count_mean_7d",
    "engagement_weight_sum_mean_7d",
    "tweet_count_zscore_24h",
    "engagement_weight_sum_zscore_24h",
    "weighted_sentiment_zscore_24h",
]
FEATURE_SETS = {
    "baseline_market_features": BASELINE_MARKET_FEATURES,
    "market_plus_sentiment_core": BASELINE_MARKET_FEATURES + SENTIMENT_CORE_FEATURES,
    "market_plus_sentiment_core_derived": (
        BASELINE_MARKET_FEATURES + SENTIMENT_CORE_FEATURES + SENTIMENT_DERIVED_FEATURES
    ),
}


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    data = pd.read_csv(dataset_path)
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], utc=True)
    data = data.sort_values(TIME_COLUMN).reset_index(drop=True)
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target column must contain at least two classes")
    return data


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> ChronologicalSplit:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Invalid chronological split ratios")

    train_end = int(len(data) * train_ratio)
    validation_end = int(len(data) * (train_ratio + validation_ratio))
    split = ChronologicalSplit(
        train=data.iloc[:train_end].copy(),
        validation=data.iloc[train_end:validation_end].copy(),
        test=data.iloc[validation_end:].copy(),
    )
    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError("Each split must contain at least one row")
    return split


def validate_feature_columns(data: pd.DataFrame, feature_columns: list[str]) -> None:
    missing_features = [column for column in feature_columns if column not in data.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    leaked_targets = sorted(set(feature_columns).intersection(TARGET_COLUMNS))
    if leaked_targets:
        raise ValueError(f"Target columns cannot be used as features: {leaked_targets}")


def split_xy(split: ChronologicalSplit, feature_columns: list[str]) -> tuple[np.ndarray, ...]:
    x_train = split.train[feature_columns].to_numpy(dtype=float)
    y_train = split.train[TARGET_COLUMN].to_numpy(dtype=int)
    x_validation = split.validation[feature_columns].to_numpy(dtype=float)
    y_validation = split.validation[TARGET_COLUMN].to_numpy(dtype=int)
    x_test = split.test[feature_columns].to_numpy(dtype=float)
    y_test = split.test[TARGET_COLUMN].to_numpy(dtype=int)
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": matrix.tolist(),
    }


def predict_scores(model: Any, x_data: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_data)
        return 1 / (1 + np.exp(-scores))
    return model.predict(x_data)


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Any:
    del x_validation, y_validation
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model.fit(x_train, y_train)


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Any:
    best_model = None
    best_score = -np.inf
    for max_depth in [3, 5, 8, None]:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        validation_score = roc_auc_score(y_validation, model.predict_proba(x_validation)[:, 1])
        if validation_score > best_score:
            best_model = model
            best_score = validation_score
    return best_model


def train_gradient_boosting(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[str, Any]:
    if importlib.util.find_spec("xgboost") is not None:
        return "XGBoost", train_xgboost(x_train, y_train, x_validation, y_validation)
    if importlib.util.find_spec("lightgbm") is not None:
        return "LightGBM", train_lightgbm(x_train, y_train, x_validation, y_validation)
    return "HistGradientBoosting", train_hist_gradient_boosting(x_train, y_train, x_validation, y_validation)


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Any:
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    try:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            early_stopping_rounds=30,
            verbose=False,
        )
    except TypeError:
        model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    return model


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    try:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
    except TypeError:
        model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)])
    return model


def train_hist_gradient_boosting(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Any:
    best_model = None
    best_score = -np.inf
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    for max_leaf_nodes in [7, 15, 31]:
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        validation_score = roc_auc_score(y_validation, model.predict_proba(x_validation)[:, 1])
        if validation_score > best_score:
            best_model = model
            best_score = validation_score
    return best_model


def train_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Any:
    del x_validation, y_validation
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model.fit(x_train, y_train)


def train_lstm_if_available(
    split: ChronologicalSplit,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    if importlib.util.find_spec("torch") is None:
        return [
            {
                "model": "LSTM",
                "status": "skipped",
                "reason": "PyTorch is not installed",
            }
        ]

    from src.models.train_sequence_models import train_lstm_models

    return train_lstm_models(split, feature_columns, TARGET_COLUMN)


def train_sklearn_models_for_feature_set(
    split: ChronologicalSplit,
    feature_set_name: str,
    feature_columns: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_feature_columns(pd.concat([split.train, split.validation, split.test]), feature_columns)
    x_train, y_train, x_validation, y_validation, x_test, y_test = split_xy(split, feature_columns)

    trainers = [
        ("LogisticRegression", train_logistic_regression),
        ("RandomForestClassifier", train_random_forest),
        ("SVM_RBF", train_svm),
    ]
    results: list[dict[str, Any]] = []
    confusion_matrices: dict[str, Any] = {}

    for model_name, trainer in trainers:
        model = trainer(x_train, y_train, x_validation, y_validation)
        y_score = predict_scores(model, x_test)
        y_pred = (y_score >= 0.5).astype(int)
        metrics = evaluate_predictions(y_test, y_pred, y_score)
        confusion_matrices[f"{feature_set_name}:{model_name}"] = metrics.pop("confusion_matrix")
        results.append(
            {
                "feature_set": feature_set_name,
                "model": model_name,
                "status": "trained",
                **metrics,
            }
        )

    boosting_name, boosting_model = train_gradient_boosting(x_train, y_train, x_validation, y_validation)
    y_score = predict_scores(boosting_model, x_test)
    y_pred = (y_score >= 0.5).astype(int)
    metrics = evaluate_predictions(y_test, y_pred, y_score)
    confusion_matrices[f"{feature_set_name}:{boosting_name}"] = metrics.pop("confusion_matrix")
    results.append(
        {
            "feature_set": feature_set_name,
            "model": boosting_name,
            "status": "trained",
            **metrics,
        }
    )
    return results, confusion_matrices


def run_model_comparison(
    dataset_path: Path,
    results_path: Path,
    confusion_path: Path,
    include_lstm: bool,
) -> pd.DataFrame:
    data = load_dataset(dataset_path)
    split = chronological_split(data)
    all_results: list[dict[str, Any]] = []
    all_confusion_matrices: dict[str, Any] = {}

    print_split_summary(split)

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        results, confusion_matrices = train_sklearn_models_for_feature_set(
            split,
            feature_set_name,
            feature_columns,
        )
        all_results.extend(results)
        all_confusion_matrices.update(confusion_matrices)

        if include_lstm and feature_set_name == "market_plus_sentiment_core_derived":
            lstm_results = train_lstm_if_available(split, feature_columns)
            for result in lstm_results:
                all_results.append({"feature_set": feature_set_name, **result})

    results = pd.DataFrame(all_results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_path.write_text(json.dumps(all_confusion_matrices, indent=2), encoding="utf-8")
    return results


def print_split_summary(split: ChronologicalSplit) -> None:
    print("Chronological split")
    for split_name, split_data in [
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ]:
        start = split_data[TIME_COLUMN].iloc[0]
        end = split_data[TIME_COLUMN].iloc[-1]
        balance = split_data[TARGET_COLUMN].value_counts(normalize=True).sort_index().to_dict()
        print(f"{split_name}: rows={len(split_data)}, range={start} to {end}, balance={balance}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model comparison across market and sentiment feature stages."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Final dataset CSV path. Default: {DEFAULT_DATASET_PATH}.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Metrics output CSV path. Default: {DEFAULT_RESULTS_PATH}.",
    )
    parser.add_argument(
        "--confusion-output",
        type=Path,
        default=DEFAULT_CONFUSION_PATH,
        help=f"Confusion matrix JSON output path. Default: {DEFAULT_CONFUSION_PATH}.",
    )
    parser.add_argument(
        "--include-lstm",
        action="store_true",
        help="Train optional LSTM comparison if PyTorch is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_model_comparison(
        args.dataset,
        args.results,
        args.confusion_output,
        args.include_lstm,
    )
    print(f"Saved model metrics to {args.results}")
    print(f"Saved confusion matrices to {args.confusion_output}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
