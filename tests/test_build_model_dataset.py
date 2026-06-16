from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.features.build_model_dataset import (
    add_leakage_safe_features,
    build_final_dataset,
    get_model_feature_columns,
)


class BuildModelDatasetTests(unittest.TestCase):
    def test_rolling_features_use_past_and_current_rows_only(self) -> None:
        data = self._base_feature_frame()
        data["weighted_sentiment_mean"] = [1.0, 2.0, 100.0]

        result = add_leakage_safe_features(data)

        self.assertEqual(result.loc[0, "weighted_sentiment_mean_24h"], 1.0)
        self.assertEqual(result.loc[1, "weighted_sentiment_mean_24h"], 1.5)
        self.assertNotEqual(result.loc[0, "weighted_sentiment_mean_24h"], (1.0 + 2.0 + 100.0) / 3)

    def test_zscore_division_by_zero_becomes_zero(self) -> None:
        data = self._base_feature_frame()
        data["tweet_count"] = [20, 20, 20]
        data["engagement_weight_sum"] = [10.0, 10.0, 10.0]
        data["weighted_sentiment_mean"] = [0.0, 0.0, 0.0]

        result = add_leakage_safe_features(data)

        zscore_columns = [
            "tweet_count_zscore_24h",
            "engagement_weight_sum_zscore_24h",
            "weighted_sentiment_zscore_24h",
        ]
        self.assertTrue((result[zscore_columns] == 0).all().all())
        self.assertFalse(result[zscore_columns].isna().any().any())

    def test_target_columns_are_excluded_from_model_features(self) -> None:
        data = self._base_feature_frame()
        feature_columns = get_model_feature_columns(data)

        self.assertNotIn("next_price_close", feature_columns)
        self.assertNotIn("next_4h_return", feature_columns)
        self.assertNotIn("target_up_next_4h", feature_columns)

    def test_output_row_count_after_target_shift_is_reasonable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            price_path = temp_path / "prices.csv"
            sentiment_path = temp_path / "sentiment.csv"
            weights_path = temp_path / "weights.csv"
            output_path = temp_path / "final.csv"

            prices = pd.DataFrame(
                {
                    "datetime_utc": pd.date_range(
                        "2025-01-01T00:00:00Z",
                        periods=12,
                        freq="h",
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "price_usd": range(100, 112),
                    "market_cap_usd": range(1000, 1012),
                    "total_volume_usd": range(2000, 2012),
                }
            )
            prices.to_csv(price_path, index=False)

            sentiment = pd.DataFrame(
                {
                    "tweet_id": [1, 2, 3],
                    "created_at": [
                        "2025-01-01T00:30:00Z",
                        "2025-01-01T04:30:00Z",
                        "2025-01-01T08:30:00Z",
                    ],
                    "sentiment_score": [0.1, -0.2, 0.3],
                    "engagement_weight": [1.0, 2.0, 3.0],
                }
            )
            sentiment.to_csv(sentiment_path, index=False)
            pd.DataFrame({"tweet_id": [1, 2, 3], "engagement_weight": [1.0, 2.0, 3.0]}).to_csv(
                weights_path,
                index=False,
            )

            build_final_dataset(price_path, sentiment_path, weights_path, output_path, 4)
            result = pd.read_csv(output_path)

            self.assertEqual(len(result), 2)
            self.assertFalse(result.isna().any().any())

    @staticmethod
    def _base_feature_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "window_start": pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="4h"),
                "price_return": [0.01, 0.02, -0.01],
                "volume_sum": [100.0, 100.0, 100.0],
                "weighted_sentiment_mean": [0.1, 0.2, 0.3],
                "sentiment_mean": [0.1, 0.2, 0.3],
                "tweet_count": [10, 20, 30],
                "engagement_weight_sum": [5.0, 10.0, 15.0],
                "next_price_close": [101.0, 102.0, 103.0],
                "next_4h_return": [0.01, 0.01, 0.01],
                "target_up_next_4h": [1, 1, 1],
            }
        )


if __name__ == "__main__":
    unittest.main()
