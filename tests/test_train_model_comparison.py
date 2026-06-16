from __future__ import annotations

import unittest

import pandas as pd

from src.models.train_model_comparison import (
    TARGET_COLUMN,
    chronological_split,
    validate_feature_columns,
)


class TrainModelComparisonTests(unittest.TestCase):
    def test_chronological_split_preserves_order(self) -> None:
        data = pd.DataFrame(
            {
                "window_start": pd.date_range("2025-01-01", periods=100, freq="4h"),
                TARGET_COLUMN: [0, 1] * 50,
            }
        )
        split = chronological_split(data)

        self.assertLess(split.train["window_start"].max(), split.validation["window_start"].min())
        self.assertLess(split.validation["window_start"].max(), split.test["window_start"].min())
        self.assertEqual(len(split.train), 70)
        self.assertEqual(len(split.validation), 15)
        self.assertEqual(len(split.test), 15)

    def test_validate_feature_columns_rejects_targets(self) -> None:
        data = pd.DataFrame(
            {
                "price_return": [0.1],
                "next_4h_return": [0.2],
                TARGET_COLUMN: [1],
            }
        )

        with self.assertRaises(ValueError):
            validate_feature_columns(data, ["price_return", "next_4h_return"])


if __name__ == "__main__":
    unittest.main()
