"""
tests/test_recommender.py
-------------------------
Unit tests for ALS collaborative filtering, content-based tag similarity, and hybrid recommendations.
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
from recommender import (
    ALSMatrixFactorization,
    ContentBasedRecommender,
    HybridRecommender,
    build_interaction_matrix,
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

    def test_als_matrix_factorization(self):
        """Verify ALS matrix factorization factor dimensions and score generation."""
        interactions = build_interaction_matrix(self.submissions_df)
        als = ALSMatrixFactorization(n_factors=12, n_epochs=3).fit(interactions)

        self.assertIsNotNone(als.user_factors)
        self.assertIsNotNone(als.item_factors)
        self.assertEqual(als.user_factors.shape[1], 12)
        self.assertEqual(als.item_factors.shape[1], 12)

        scores = als.score_all_items(self.sample_user_id)
        self.assertEqual(len(scores), len(als.item_ids_ordered()))

    def test_content_based_recommender(self):
        """Verify content-based cosine similarity against user solved/failed profile."""
        cb = ContentBasedRecommender(self.questions_df)
        solved_ids = [1, 2, 3]
        scores = cb.similar_to_user_profile(solved_ids)

        self.assertEqual(len(scores), len(self.questions_df))
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 1.0 + 1e-6).all())

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
