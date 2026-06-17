"""Training loop with early stopping, LR reduction, and optional model checkpointing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential


def train(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    reduce_lr_patience: int = 7,
    model_save_path: str | Path | None = None,
    verbose: int = 1,
    class_weight: dict | None = None,
):
    """Train a Keras model with standard callbacks.

    Args:
        model:            compiled Keras model
        X_train, y_train: training sequences and labels
        X_val,   y_val:   validation sequences and labels
        epochs:           max training epochs
        batch_size:       mini-batch size
        patience:         EarlyStopping patience (val_loss)
        reduce_lr_patience: ReduceLROnPlateau patience
        model_save_path:  if given, saves the best model weights here
        verbose:          Keras verbosity (0=silent, 1=progress, 2=epoch)

    Returns:
        Keras History object
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=reduce_lr_patience,
            factor=0.25,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if model_save_path is not None:
        save_path = Path(model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                str(save_path),
                save_best_only=True,
                monitor="val_loss",
                verbose=0,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        class_weight=class_weight,
    )
    return history
