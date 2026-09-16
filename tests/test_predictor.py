"""
tests/test_predictor.py
-----------------------
Unit tests for the XGBoost contest rating regression engine, offline training,
model serialization, and single-user inference without retraining.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

import numpy as np
import pandas as pd

import config
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
        self.assertGreater(metrics["r2"], 0.35)

        importances = predictor.feature_importance()
        self.assertIsInstance(importances, pd.DataFrame)
        self.assertIn("feature", importances.columns)
        self.assertIn("importance", importances.columns)

    def test_insufficient_samples_guard(self):
        """Verify model rejects fitting on <10 samples and does not mark as trained."""
        predictor = ContestRatingPredictor()
        single_user_matrix = self.feature_matrix.head(1)

        metrics = predictor.fit(single_user_matrix)
        self.assertFalse(metrics["available"])
        self.assertFalse(predictor.is_trained)
        self.assertIn("multi-user", metrics["note"].lower())

    def test_save_and_load_model(self):
        """Verify model serialization and deserialization integrity."""
        predictor = ContestRatingPredictor()
        predictor.fit(self.feature_matrix)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            predictor.save_model(tmp_path)
            self.assertTrue(tmp_path.exists())

            loaded_predictor = ContestRatingPredictor().load_model(tmp_path)
            self.assertTrue(loaded_predictor.is_trained)
            self.assertEqual(loaded_predictor.feature_columns_, predictor.feature_columns_)

            test_sample = self.feature_matrix.head(5)
            preds_orig = predictor.predict(test_sample)
            preds_loaded = loaded_predictor.predict(test_sample)
            np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_single_user_inference_without_retraining(self):
        """Verify pre-trained model infers on single real-user row without retraining."""
        predictor = ContestRatingPredictor()
        if config.RATING_MODEL_PATH.exists():
            predictor.load_model(config.RATING_MODEL_PATH)
        else:
            predictor.fit(self.feature_matrix)

        initial_model_obj = predictor.pipeline.named_steps["model"]

        single_user = self.feature_matrix.iloc[[0]].copy()
        pred = predictor.predict(single_user)

        self.assertEqual(len(pred), 1)
        self.assertGreaterEqual(pred[0], 800.0)
        self.assertLessEqual(pred[0], 3500.0)

        # Model pipeline must NOT be re-instantiated or refitted
        self.assertIs(predictor.pipeline.named_steps["model"], initial_model_obj)

    def test_no_target_leakage(self):
        """Verify latent simulation variables and target rating are never in model features."""
        predictor = ContestRatingPredictor()
        predictor.fit(self.feature_matrix)

        forbidden_features = {"latent_skill", "contest_rating", "archetype"}
        actual_features = set(predictor.feature_columns_)
        self.assertSetEqual(actual_features.intersection(forbidden_features), set())


if __name__ == "__main__":
    unittest.main()
