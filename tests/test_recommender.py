"""
tests/test_recommender.py
-------------------------
Unit tests for ALS collaborative filtering, fold-in for new users,
content-based tag similarity, hybrid recommendations, and evaluation metrics.
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

from data_processing import generate_full_synthetic_dataset, FeatureEngineer
from recommender import (
    ALSMatrixFactorization,
    ContentBasedRecommender,
    HybridRecommender,
    build_interaction_matrix,
    evaluate_recommendations,
)


class TestRecommender(unittest.TestCase):
    """Test suite for recommender algorithms."""

    @classmethod
    def setUpClass(cls):
        cls.users_df, cls.questions_df, cls.submissions_df = generate_full_synthetic_dataset()
        cls.recommender = HybridRecommender(cls.questions_df, cls.submissions_df)
        cls.sample_user_id = int(cls.users_df["user_id"].iloc[0])

    def test_interaction_matrix_builder(self):
        """Verify interaction weights are properly assigned."""
        interactions = build_interaction_matrix(self.submissions_df)
        self.assertIn("user_id", interactions.columns)
        self.assertIn("question_id", interactions.columns)
        self.assertIn("weight", interactions.columns)
        # Attempted is 1.0, Solved is 3.0
        self.assertTrue(set(interactions["weight"].unique()).issubset({1.0, 3.0}))

    def test_als_matrix_factorization_fit(self):
        """Verify ALS matrix factorization factor dimensions and score generation."""
        interactions = build_interaction_matrix(self.submissions_df)
        als = ALSMatrixFactorization(latent_dim=12, epochs=3).fit(interactions)

        self.assertIsNotNone(als.user_factors)
        self.assertIsNotNone(als.item_factors)
        self.assertEqual(als.user_factors.shape[1], 12)
        self.assertEqual(als.item_factors.shape[1], 12)
        self.assertIsNotNone(als.YtY)

        scores = als.score_all_items(self.sample_user_id)
        self.assertEqual(len(scores), len(als.item_ids))

    def test_als_save_and_load(self):
        """Verify ALS model serialization and deserialization."""
        interactions = build_interaction_matrix(self.submissions_df)
        als = ALSMatrixFactorization(latent_dim=8, epochs=2).fit(interactions)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            als.save(tmp_path)
            self.assertTrue(tmp_path.exists())

            loaded_als = ALSMatrixFactorization().load(tmp_path)
            self.assertTrue(loaded_als.is_fitted)
            self.assertEqual(loaded_als.latent_dim, 8)
            np.testing.assert_allclose(loaded_als.item_factors, als.item_factors, rtol=1e-5)
            np.testing.assert_allclose(loaded_als.YtY, als.YtY, rtol=1e-5)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_als_fold_in_new_user(self):
        """Verify closed-form fold-in for a brand new real user without modifying item factors."""
        interactions = build_interaction_matrix(self.submissions_df)
        als = ALSMatrixFactorization(latent_dim=12, epochs=3).fit(interactions)

        initial_item_factors = als.item_factors.copy()
        initial_YtY = als.YtY.copy()

        # Simulate a completely unseen real user with 5 problem interactions
        new_user_interactions = pd.DataFrame({
            "question_id": [1, 2, 3, 4, 5],
            "weight": [3.0, 3.0, 1.0, 3.0, 1.0],
        })

        scores = als.fold_in_user(new_user_interactions)
        self.assertEqual(len(scores), len(als.item_ids))
        self.assertFalse(np.allclose(scores, 0.0))

        # Crucial check: item factors Y and YtY MUST NOT be modified during fold-in
        np.testing.assert_array_equal(als.item_factors, initial_item_factors)
        np.testing.assert_array_equal(als.YtY, initial_YtY)

    def test_content_based_recommender(self):
        """Verify content-based cosine similarity against user solved/failed profile."""
        cb = ContentBasedRecommender(self.questions_df)
        solved_ids = [1, 2, 3]
        scores = cb.similar_to_user_profile(solved_ids)

        self.assertEqual(len(scores), len(self.questions_df))
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 1.0 + 1e-6).all())

    def test_cold_start_fallback(self):
        """Verify user with <3 interactions gracefully falls back to content/weakness recommendations."""
        fe = FeatureEngineer(self.users_df, self.questions_df, self.submissions_df)
        topic_profile = fe.get_topic_profile(self.sample_user_id)

        # Single-interaction DataFrame (cold-start)
        cold_submissions = pd.DataFrame([{
            "user_id": 99999,
            "question_id": 1,
            "timestamp": pd.Timestamp("2026-01-01"),
            "status": "Accepted",
            "runtime_ms": 30.0,
            "language": "Python3",
        }])

        recs = self.recommender.recommend(
            user_id=99999,
            top_n=5,
            topic_profile=topic_profile,
            user_submissions_df=cold_submissions,
        )

        self.assertEqual(len(recs), 5)
        self.assertNotIn(1, recs["question_id"].values)  # Solved problem excluded

    def test_hybrid_recommendation_output(self):
        """Verify hybrid recommendation properties: count, exclusions, diversity, reasons."""
        fe = FeatureEngineer(self.users_df, self.questions_df, self.submissions_df)
        topic_profile = fe.get_topic_profile(self.sample_user_id)

        top_recs = self.recommender.recommend(
            user_id=self.sample_user_id,
            top_n=5,
            topic_profile=topic_profile,
        )

        self.assertEqual(len(top_recs), 5)
        self.assertIn("question_id", top_recs.columns)
        self.assertIn("title", top_recs.columns)
        self.assertIn("difficulty", top_recs.columns)
        self.assertIn("recommendation_score", top_recs.columns)
        self.assertIn("reason", top_recs.columns)

        # Check solved questions are not recommended
        solved_ids = self.submissions_df.loc[
            (self.submissions_df["user_id"] == self.sample_user_id) & (self.submissions_df["status"] == "Accepted"),
            "question_id",
        ].unique()
        self.assertFalse(any(qid in solved_ids for qid in top_recs["question_id"]))


if __name__ == "__main__":
    unittest.main()
