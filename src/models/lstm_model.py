"""LSTM model architecture and data preparation for financial time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------------------------
# Sequence preparation
# ---------------------------------------------------------------------------

def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of length look_back over features and pair with the target at each step."""
    X, y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back : i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    look_back: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scale_target: bool = False,
) -> tuple:
    """Scale features, build sequences, and split chronologically.

    Args:
        scale_target: if True, also fit a separate MinMaxScaler on the target
                      column (required for regression so predictions/losses are
                      on the same [0,1] scale as features).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler
        target_scaler is None when scale_target=False.
    """
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        print(f"[prepare_data] Skipping missing columns: {missing}")

    df_clean = df[available + [target_col]].dropna().copy()
    n = len(df_clean)
    if n <= look_back:
        raise ValueError(f"Not enough rows ({n}) for look_back={look_back}.")

    feature_matrix = df_clean[available].values
    target_vector = df_clean[target_col].values.astype(float)

    # Fit scalers only on training portion to prevent data leakage
    train_end = int(n * (1 - val_ratio - test_ratio))
    scaler = MinMaxScaler()
    scaler.fit(feature_matrix[:train_end])
    feature_scaled = scaler.transform(feature_matrix)

    target_scaler = None
    if scale_target:
        target_scaler = MinMaxScaler()
        target_scaler.fit(target_vector[:train_end].reshape(-1, 1))
        target_vector = target_scaler.transform(target_vector.reshape(-1, 1)).flatten()

    X, y = create_sequences(feature_scaled, target_vector, look_back)
    total = len(X)

    test_n = int(total * test_ratio)
    val_n = int(total * val_ratio)
    train_n = total - val_n - test_n

    X_train, y_train = X[:train_n], y[:train_n]
    X_val,   y_val   = X[train_n : train_n + val_n], y[train_n : train_n + val_n]
    X_test,  y_test  = X[train_n + val_n :], y[train_n + val_n :]

    print(
        f"[prepare_data] train={X_train.shape} | val={X_val.shape} | test={X_test.shape} "
        f"| features={len(available)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_scaler


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_lstm(
    input_shape: tuple[int, int],
    units: list[int] | None = None,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    task: str = "classification",
) -> Sequential:
    """Build a stacked LSTM model.

    Args:
        input_shape: (look_back, n_features)
        units:       list of unit counts per LSTM layer, e.g. [128, 64]
        dropout:     dropout fraction after each LSTM layer
        learning_rate: Adam learning rate
        task:        "classification" (sigmoid + BCE) or "regression" (linear + MSE)
    """
    if units is None:
        units = [64, 32]

    model = Sequential()
    model.add(Input(shape=input_shape))

    for i, n_units in enumerate(units):
        return_sequences = i < len(units) - 1
        model.add(LSTM(n_units, return_sequences=return_sequences))
        model.add(Dropout(dropout))

    # Single dense head — no intermediate layer to avoid over-regularising small datasets
    if task == "classification":
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model.add(Dense(1, activation="linear"))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )

    return model
