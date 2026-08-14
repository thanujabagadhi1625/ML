"""
tests/test_predictor.py
-----------------------
Unit tests for the XGBoost contest rating regression engine and single-user fallback.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

import numpy as np
import pandas as pd

from data_processing import generate_full_synthetic_dataset, FeatureEngineer
from predictor import ContestRatingPredictor


class TestPredictor(unittest.TestCase):
    """Test suite for rating regression pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.users_df, cls.questions_df, cls.submissions_df = generate_full_synthetic_dataset()
        fe = FeatureEngineer(cls.users_df, cls.questions_df, cls.submissions_df)
        cls.feature_matrix = fe.build_user_feature_matrix()

    def test_xgboost_fit_and_metrics(self):
        """Verify model fitting on multi-user matrix returns valid regression metrics."""
        predictor = ContestRatingPredictor()
        metrics = predictor.fit(self.feature_matrix)

        self.assertTrue(metrics["available"])
        self.assertIsNotNone(metrics["rmse"])
        self.assertIsNotNone(metrics["mae"])
        self.assertIsNotNone(metrics["r2"])
        self.assertGreater(metrics["r2"], 0.5)  # Should achieve strong fit on synthetic ground truth

        importances = predictor.feature_importance()
        self.assertIsInstance(importances, pd.DataFrame)
        self.assertIn("feature", importances.columns)
        self.assertIn("importance", importances.columns)

    def test_single_user_fallback(self):
        """Verify graceful fallback when dataset contains insufficient multi-user samples."""
        predictor = ContestRatingPredictor()
        single_user_matrix = self.feature_matrix.head(1)

        metrics = predictor.fit(single_user_matrix)
        self.assertFalse(metrics["available"])
        self.assertIn("insufficient", metrics["note"].lower())

        # Prediction should use heuristic fallback
        predictions = predictor.predict(single_user_matrix)
        self.assertEqual(len(predictions), 1)
        self.assertGreaterEqual(predictions[0], 800.0)
        self.assertLessEqual(predictions[0], 3000.0)


if __name__ == "__main__":
    unittest.main()
