"""
tests/test_audit_fixes.py
-------------------------
Unit tests validating fixes for all audit findings:
  1. Double user mapping bug in load_leetcode_history_export and load_leetcode_history_records.
  2. Frontend LeetCode ID numeric matching in _build_questions_from_export.
  3. Catalogue item completeness in ALSMatrixFactorization.fit (all_item_ids).
  4. Train/serve consistency for recency_momentum calculation across batch and single-user.
  5. Fold-in confidence on raw submissions and repeated attempt capping.
  6. Cold-start honest reason text in recommend_two_lists.
  7. Custom/unseen question metadata preservation in FeatureEngineer.__init__.
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

import config
from data_processing import (
    FeatureEngineer,
    _build_questions_from_export,
    _normalize_export_submissions,
    load_canonical_questions,
    load_leetcode_history_records,
)
from recommender import (
    ALSMatrixFactorization,
    HybridRecommender,
    build_interaction_matrix,
)


class TestAuditFixes(unittest.TestCase):
    """Test suite covering audit bug fixes and train/serve invariants."""

    def test_export_user_id_mapping_preserves_non_one_ids(self):
        """Fix 1: Submissions with arbitrary user_id (e.g. 42) must not produce NaN user_id."""
        records = [
            {
                "user_id": 42,
                "question_id": "two-sum",
                "title": "Two Sum",
                "difficulty": "Easy",
                "status": "Accepted",
                "timestamp": 1700000000,
            },
            {
                "user_id": 99,
                "question_id": "add-two-numbers",
                "title": "Add Two Numbers",
                "difficulty": "Medium",
                "status": "Accepted",
                "timestamp": 1700001000,
            },
        ]
        users_df, questions_df, subs_df = load_leetcode_history_records(records)
        self.assertFalse(users_df["user_id"].isna().any(), "users_df contains NaN user_id")
        self.assertFalse(subs_df["user_id"].isna().any(), "subs_df contains NaN user_id")
        self.assertEqual(len(users_df), 2)
        self.assertListEqual(sorted(users_df["user_id"].tolist()), [1, 2])
        self.assertListEqual(sorted(subs_df["user_id"].unique().tolist()), [1, 2])

    def test_build_questions_matches_frontend_leetcode_id(self):
        """Fix 2: Numeric question_id='1' in exports must match Two Sum (leetcode_id=1)."""
        df = pd.DataFrame([{
            "user_id": 1,
            "question_id": "1",
            "title": "1",
            "difficulty": "Easy",
            "status": "Accepted",
            "timestamp": "2026-01-01",
        }])
        norm_df = _normalize_export_submissions(df)
        full_q, qid_map = _build_questions_from_export(norm_df)
        mapped_id = qid_map["1"]
        matched_row = full_q[full_q["question_id"] == mapped_id].iloc[0]
        self.assertEqual(matched_row["slug"], "two-sum")

    def test_als_fit_with_all_item_ids_indexes_full_catalogue(self):
        """Fix 3: All catalogue items must exist in item_factors even if some had 0 training interactions."""
        interactions = pd.DataFrame({
            "user_id": [1, 1, 2],
            "question_id": [10, 20, 10],
            "weight": [3.0, 1.0, 3.0],
        })
        all_ids = [10, 20, 30, 40, 50]
        als = ALSMatrixFactorization(n_factors=4, n_epochs=5, random_state=42)
        als.fit(interactions, all_item_ids=all_ids)
        for qid in all_ids:
            self.assertIn(qid, als.item_index_, f"Question {qid} missing from ALS item_index_")
        self.assertEqual(als.item_factors.shape[0], len(all_ids))
        self.assertEqual(len(als.item_index_), len(all_ids))

    def test_recency_momentum_batch_vs_single_user_invariance(self):
        """Fix 4: Momentum for a single user must be invariant whether evaluated alone or inside a batch."""
        canonical_q = load_canonical_questions()
        user_a_subs = pd.DataFrame({
            "user_id": [1, 1],
            "question_id": [canonical_q.iloc[0]["question_id"], canonical_q.iloc[1]["question_id"]],
            "timestamp": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10")],
            "status": ["Wrong Answer", "Accepted"],
            "runtime_ms": [50.0, 45.0],
            "language": ["Python3", "Python3"],
        })
        user_b_subs = pd.DataFrame({
            "user_id": [2, 2],
            "question_id": [canonical_q.iloc[2]["question_id"], canonical_q.iloc[3]["question_id"]],
            "timestamp": [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-15")],
            "status": ["Accepted", "Accepted"],
            "runtime_ms": [60.0, 55.0],
            "language": ["Python3", "Python3"],
        })

        # Single user evaluation
        u1_df = pd.DataFrame([{"user_id": 1, "account_age_days": 100, "latent_skill": 0.0, "contest_rating": 1500.0}])
        fe_single = FeatureEngineer(u1_df, canonical_q, user_a_subs)
        feat_single = fe_single.build_user_feature_matrix().set_index("user_id")
        mom_single = feat_single.loc[1, "recency_momentum"]

        # Batch evaluation
        u_batch_df = pd.DataFrame([
            {"user_id": 1, "account_age_days": 100, "latent_skill": 0.0, "contest_rating": 1500.0},
            {"user_id": 2, "account_age_days": 200, "latent_skill": 0.0, "contest_rating": 1600.0},
        ])
        subs_batch = pd.concat([user_a_subs, user_b_subs], ignore_index=True)
        fe_batch = FeatureEngineer(u_batch_df, canonical_q, subs_batch)
        feat_batch = fe_batch.build_user_feature_matrix().set_index("user_id")
        mom_batch = feat_batch.loc[1, "recency_momentum"]

        self.assertAlmostEqual(mom_single, mom_batch, places=5,
                               msg="Train/serve mismatch: recency_momentum differs when evaluated in batch!")

    def test_fold_in_raw_submissions_and_repeat_capping(self):
        """Fix 5: fold_in_user on raw submissions DataFrame without 'weight' computes weights and caps duplicates."""
        canonical_q = load_canonical_questions()
        q1 = canonical_q.iloc[0]["question_id"]
        q2 = canonical_q.iloc[1]["question_id"]
        q3 = canonical_q.iloc[2]["question_id"]

        train_inter = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3],
            "question_id": [q1, q2, q2, q3, q1, q3],
            "weight": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        })
        als = ALSMatrixFactorization(n_factors=8, n_epochs=5, random_state=42)
        als.fit(train_inter, all_item_ids=[q1, q2, q3])

        raw_subs = pd.DataFrame({
            "user_id": [99] * 5,
            "question_id": [q1, q1, q2, q3, q3],
            "status": ["Wrong Answer", "Accepted", "Accepted", "Wrong Answer", "Accepted"],
            "timestamp": [pd.Timestamp("2026-01-01")] * 5,
        })
        scores = als.fold_in_user(raw_subs)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(als.item_index_))
        self.assertTrue(np.any(scores != 0.0))

    def test_recommend_two_lists_cold_start_reason(self):
        """Fix 6: Cold start users (< 3 interactions) must not claim collaborative filtering rationale."""
        canonical_q = load_canonical_questions()
        recommender = HybridRecommender(canonical_q)
        empty_subs = pd.DataFrame(columns=["user_id", "question_id", "timestamp", "status"])
        res = recommender.recommend_two_lists(user_id=1, user_submissions_df=empty_subs, top_n=5)
        pop_df = res["popular_next_problems"]
        self.assertFalse(pop_df.empty)
        for _, row in pop_df.iterrows():
            reason = row["reason"]
            self.assertNotIn("Collaborative filtering candidate", reason,
                             "Cold-start recommendation falsely claimed collaborative filtering!")

    def test_feature_engineer_preserves_custom_question_difficulty(self):
        """Fix 7: Submissions for questions not in canonical catalogue retain submitted difficulty."""
        u_df = pd.DataFrame([{"user_id": 1, "account_age_days": 50, "latent_skill": 0.0, "contest_rating": 1500.0}])
        subs = pd.DataFrame([{
            "user_id": 1,
            "question_id": 999999,  # unseen question
            "timestamp": pd.Timestamp("2026-01-01"),
            "status": "Accepted",
            "difficulty": "Hard",
            "runtime_ms": 40.0,
            "language": "Python3",
        }])
        canonical_q = load_canonical_questions()
        fe = FeatureEngineer(u_df, canonical_q, subs)
        acc_ratios = fe.compute_accuracy_ratios()
        self.assertEqual(acc_ratios.loc[1, "accuracy_hard"], 1.0)
        diff_dist = fe.compute_difficulty_distribution()
        self.assertEqual(diff_dist.loc[1, "share_hard"], 1.0)


if __name__ == "__main__":
    unittest.main()
