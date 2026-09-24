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

    def test_no_latent_columns_in_feature_matrix(self):
        """Verify strictly no latent or generator-only columns enter the feature matrix."""
        users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
        fe = FeatureEngineer(users_df, questions_df, submissions_df)
        fm = fe.build_user_feature_matrix()

        forbidden_cols = [
            "latent_skill", "archetype", "base_skill", "hard_resilience",
            "learning_slope", "attempt_multiplier", "topic_affinities",
            "skill_level", "initial_skill", "archetype_name", "user_archetype",
            "topic_weights", "latent_topics", "user_latent_vector",
        ]
        for col in forbidden_cols:
            self.assertNotIn(col, fm.columns)

        # Also test with generator v3 and intentional leakage injection
        from data_processing import generate_synthetic_dataset_v3
        u3, q3, s3, l3 = generate_synthetic_dataset_v3(questions_df, n_users=20, random_seed=42)
        u3["latent_skill"] = 1.5
        u3["archetype"] = "specialized"
        u3["base_skill"] = 2.0
        fe3 = FeatureEngineer(u3, q3, s3)
        fm3 = fe3.build_user_feature_matrix()
        for col in forbidden_cols:
            self.assertNotIn(col, fm3.columns)
        self.assertIn("account_age_days", fm3.columns)

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

    def test_user_archetypes_generation(self):
        """Verify synthetic generation with 9 archetypes produces expected distributions."""
        users_df, latent_df = generate_synthetic_users(n=90, return_latent_info=True)
        self.assertEqual(len(users_df), 90)
        self.assertIn("archetype", latent_df.columns)
        unique_archetypes = latent_df["archetype"].unique()
        self.assertGreaterEqual(len(unique_archetypes), 5)

    def test_user_level_splitting_no_leakage(self):
        """Verify user_train_val_test_split has strictly zero user overlap across splits."""
        from data_processing import user_train_val_test_split
        users_df, questions_df, submissions_df = generate_full_synthetic_dataset()

        train_u, val_u, test_u, train_s, val_s, test_s = user_train_val_test_split(
            users_df, submissions_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        train_set = set(train_u["user_id"])
        val_set = set(val_u["user_id"])
        test_set = set(test_u["user_id"])

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

        # Verify submission user IDs strictly match their user split
        self.assertTrue(set(train_s["user_id"]).issubset(train_set))
        self.assertTrue(set(val_s["user_id"]).issubset(val_set))
        self.assertTrue(set(test_s["user_id"]).issubset(test_set))

    def test_temporal_splitting_no_future_leakage(self):
        """Verify temporal_split_user_submissions partitions chronologically without future leakage."""
        from data_processing import temporal_split_user_submissions
        users_df, questions_df, submissions_df = generate_full_synthetic_dataset()

        sample_uid = int(users_df["user_id"].iloc[0])
        user_subs = submissions_df[submissions_df["user_id"] == sample_uid].copy()

        if len(user_subs) >= 5:
            hist_df, holdout_df = temporal_split_user_submissions(user_subs, history_ratio=0.8)
            self.assertGreater(len(hist_df), 0)
            self.assertGreater(len(holdout_df), 0)
            self.assertEqual(len(hist_df) + len(holdout_df), len(user_subs))

            # Max timestamp in history must be <= min timestamp in holdout
            max_hist_time = pd.to_datetime(hist_df["timestamp"]).max()
            min_holdout_time = pd.to_datetime(holdout_df["timestamp"]).min()
            self.assertLessEqual(max_hist_time, min_holdout_time)

    def test_declining_momentum_rule(self):
        """Verify declining momentum rule requires >= 8 recent attempts and >= 10 pp drop."""
        from data_processing import check_declining_momentum

        # 69% -> 68% (drop = 1 pp < 10 pp): not declining
        self.assertFalse(check_declining_momentum(0.69, 0.68, recent_attempts=10))
        self.assertFalse(check_declining_momentum(69, 68, recent_attempts=10))
        self.assertFalse(check_declining_momentum(0.69, 0.68, recent_attempts=8))

        # 71% -> 56% with 10 attempts (drop = 15 pp >= 10 pp, attempts = 10 >= 8): declining
        self.assertTrue(check_declining_momentum(0.71, 0.56, recent_attempts=10))
        self.assertTrue(check_declining_momentum(71, 56, recent_attempts=10))

        # 71% -> 56% with 7 attempts (drop = 15 pp >= 10 pp, but attempts = 7 < 8): not declining
        self.assertFalse(check_declining_momentum(0.71, 0.56, recent_attempts=7))
        self.assertFalse(check_declining_momentum(71, 56, recent_attempts=0))

    def test_difficulty_column_shares(self):
        """Verify Difficulty column computes share of Easy/Medium/Hard attempts (e.g. E20% M65% H15%)."""
        from data_processing import compute_user_topic_profile

        # 20 Easy, 65 Medium, 15 Hard = 100 attempts -> "E20% M65% H15%"
        q_records = []
        for i in range(20):
            q_records.append({"question_id": 100 + i, "difficulty": "Easy", "topic_tags": ["Array"]})
        for i in range(65):
            q_records.append({"question_id": 200 + i, "difficulty": "Medium", "topic_tags": ["Array"]})
        for i in range(15):
            q_records.append({"question_id": 300 + i, "difficulty": "Hard", "topic_tags": ["Array"]})
        q_df = pd.DataFrame(q_records)

        s_records = []
        for i, row in enumerate(q_records):
            s_records.append({
                "submission_id": i + 1,
                "user_id": 1,
                "question_id": row["question_id"],
                "status": "Accepted" if i % 2 == 0 else "Wrong Answer",
                "difficulty": row["difficulty"],
                "topic_tags": row["topic_tags"],
            })
        s_df = pd.DataFrame(s_records)

        profile = compute_user_topic_profile(s_df, q_df, user_id=1)
        self.assertFalse(profile.empty)
        array_row = profile[profile["topic"] == "Array"].iloc[0]
        self.assertEqual(array_row["dominant_difficulty"], "E20% M65% H15%")
        self.assertEqual(array_row["difficulty_share"], "E20% M65% H15%")


if __name__ == "__main__":
    unittest.main()


