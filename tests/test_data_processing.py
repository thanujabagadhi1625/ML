"""
tests/test_data_processing.py
-----------------------------
Unit tests for data generation, normalization, topic taxonomy, and feature engineering.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

# Ensure files directory is in python path
FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

import numpy as np
import pandas as pd

from data_processing import (
    FeatureEngineer,
    _parse_topic_tags,
    generate_full_synthetic_dataset,
    generate_synthetic_questions,
    generate_synthetic_submissions,
    generate_synthetic_users,
    load_leetcode_history_records,
)
from schemas import QUESTION_COLUMNS, SUBMISSION_COLUMNS, USER_COLUMNS


class TestDataProcessing(unittest.TestCase):
    """Test suite for data generation and feature engineering."""

    def test_synthetic_data_generation(self):
        """Verify synthetic tables conform to typed schema contracts."""
        questions_df = generate_synthetic_questions(n=50)
        self.assertEqual(len(questions_df), 50)
        self.assertListEqual(list(questions_df.columns), QUESTION_COLUMNS)

        users_df = generate_synthetic_users(n=20)
        self.assertEqual(len(users_df), 20)
        self.assertListEqual(list(users_df.columns), USER_COLUMNS)

        submissions_df = generate_synthetic_submissions(users_df, questions_df, n_submissions=200)
        self.assertEqual(len(submissions_df), 200)
        self.assertListEqual(list(submissions_df.columns), SUBMISSION_COLUMNS)

    def test_topic_tag_parsing(self):
        """Verify canonical topic taxonomy parsing and noise filtering."""
        # Exact canonical tag
        self.assertEqual(_parse_topic_tags(["Dynamic Programming", "Graph"]), ["Dynamic Programming", "Graph"])

        # String representation of list
        self.assertEqual(_parse_topic_tags("['Binary Search', 'Tree']"), ["Binary Search", "Tree"])

        # Inferred/fuzzy matching
        self.assertEqual(_parse_topic_tags("dp, graph traversal"), ["Dynamic Programming", "Graph"])

        # Non-canonical noise filtered out
        self.assertEqual(_parse_topic_tags(["Bananas", "Candies", "Arrays"]), ["Array"])

    def test_feature_engineering_pipeline(self):
        """Verify feature matrix computation and recency momentum decay."""
        users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
        fe = FeatureEngineer(users_df, questions_df, submissions_df)

        feature_matrix = fe.build_user_feature_matrix()
        self.assertIn("user_id", feature_matrix.columns)
        self.assertIn("overall_accuracy", feature_matrix.columns)
        self.assertIn("recency_momentum", feature_matrix.columns)
        self.assertNotIn("latent_skill", feature_matrix.columns)  # Ground truth must not leak

        # Check values are in valid ranges
        self.assertTrue((feature_matrix["overall_accuracy"] >= 0.0).all())
        self.assertTrue((feature_matrix["overall_accuracy"] <= 1.0).all())
        self.assertTrue((feature_matrix["recency_momentum"] >= 0.0).all())
        self.assertTrue((feature_matrix["recency_momentum"] <= 1.0).all())

    def test_topic_weakness_profile(self):
        """Verify structured topic weakness profile generation."""
        users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
        fe = FeatureEngineer(users_df, questions_df, submissions_df)

        sample_uid = int(users_df["user_id"].iloc[0])
        profile = fe.get_topic_profile(user_id=sample_uid)

        self.assertIsInstance(profile, pd.DataFrame)
        if not profile.empty:
            self.assertIn("topic", profile.columns)
            self.assertIn("attempts", profile.columns)
            self.assertIn("success_rate_pct", profile.columns)
            self.assertIn("weakness_level", profile.columns)
            self.assertIn("risk_score", profile.columns)


if __name__ == "__main__":
    unittest.main()
