"""Optional PyTorch LSTM sequence models for comparison only."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def train_lstm_models(split: Any, feature_columns: list[str], target_column: str) -> list[dict[str, Any]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    results: list[dict[str, Any]] = []
    scaler = StandardScaler()
    x_train_rows = scaler.fit_transform(split.train[feature_columns].to_numpy(dtype=float))
    x_validation_rows = scaler.transform(split.validation[feature_columns].to_numpy(dtype=float))
    x_test_rows = scaler.transform(split.test[feature_columns].to_numpy(dtype=float))
    y_train_rows = split.train[target_column].to_numpy(dtype=float)
    y_validation_rows = split.validation[target_column].to_numpy(dtype=float)
    y_test_rows = split.test[target_column].to_numpy(dtype=float)

    for sequence_length in [6, 12]:
        train_dataset = make_sequence_dataset(x_train_rows, y_train_rows, sequence_length)
        validation_dataset = make_sequence_dataset(x_validation_rows, y_validation_rows, sequence_length)
        test_dataset = make_sequence_dataset(x_test_rows, y_test_rows, sequence_length)
        if min(len(train_dataset[0]), len(validation_dataset[0]), len(test_dataset[0])) == 0:
            results.append(
                {
                    "model": f"LSTM_seq{sequence_length}",
                    "status": "skipped",
                    "reason": "not enough rows for sequence length",
                }
            )
            continue

        for hidden_size in [16, 32]:
            model = LSTMClassifier(
                input_size=x_train_rows.shape[1],
                hidden_size=hidden_size,
                dropout=0.2,
            )
            trained_model = fit_lstm(
                model,
                train_dataset,
                validation_dataset,
                max_epochs=50,
                batch_size=32,
            )
            metrics = evaluate_lstm(trained_model, test_dataset)
            results.append(
                {
                    "model": f"LSTM_seq{sequence_length}_hidden{hidden_size}",
                    "status": "trained",
                    **metrics,
                }
            )
    return results


class LSTMClassifier:
    def __new__(cls, input_size: int, hidden_size: int, dropout: float) -> Any:
        import torch
        from torch import nn

        class _LSTMClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.output = nn.Linear(hidden_size, 1)

            def forward(self, x_data: torch.Tensor) -> torch.Tensor:
                _, (hidden, _) = self.lstm(x_data)
                hidden = self.dropout(hidden[-1])
                return self.output(hidden).squeeze(1)

        return _LSTMClassifier()


def make_sequence_dataset(
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for index in range(sequence_length - 1, len(x_rows)):
        sequences.append(x_rows[index - sequence_length + 1 : index + 1])
        targets.append(y_rows[index])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def fit_lstm(
    model: Any,
    train_dataset: tuple[np.ndarray, np.ndarray],
    validation_dataset: tuple[np.ndarray, np.ndarray],
    max_epochs: int,
    batch_size: int,
) -> Any:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    x_train, y_train = train_dataset
    x_validation, y_validation = validation_dataset
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=False,
    )
    validation_x = torch.from_numpy(x_validation)
    validation_y = torch.from_numpy(y_validation)
    best_state = None
    best_loss = float("inf")
    patience = 5
    stale_epochs = 0

    for _ in range(max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(validation_x), validation_y).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_lstm(
    model: Any,
    test_dataset: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    import torch

    x_test, y_test = test_dataset
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test))
        y_score = torch.sigmoid(logits).numpy()
    y_pred = (y_score >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    try:
        roc_auc = roc_auc_score(y_test, y_score)
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": matrix.tolist(),
    }
