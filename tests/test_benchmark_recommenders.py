"""
tests/test_benchmark_recommenders.py
-----------------------------------
Unit tests for the extended benchmark recommender harness:
- 8 evaluation methods
- Product-goal metrics (weak_coverage@K, difficulty_fit@K)
- User difficulty level calculation
- Hybrid diversity flag and popularity blending
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
TRAINING_DIR = ROOT_DIR / "training"
for p in [str(FILES_DIR), str(TRAINING_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_recommenders import (
    compute_user_difficulty_level,
    get_popularity_counts,
    get_popularity_rankings,
    run_benchmark,
)
from recommender import (
    ALSMatrixFactorization,
    ContentBasedRecommender,
    HybridRecommender,
)


class TestBenchmarkHarness(unittest.TestCase):
    def setUp(self):
        self.qid_to_diff = {
            1: "Easy",
            2: "Easy",
            3: "Medium",
            4: "Medium",
            5: "Hard",
            6: "Hard",
        }

    def test_compute_user_difficulty_level_empty(self):
        empty_df = pd.DataFrame(columns=["question_id", "status", "difficulty"])
        self.assertEqual(compute_user_difficulty_level(empty_df, self.qid_to_diff), 0)

    def test_compute_user_difficulty_level_hierarchy(self):
        # User has >= 3 attempts on Medium with 66% acc, but only 2 attempts on Hard (50% acc)
        subs = pd.DataFrame([
            {"question_id": 3, "status": "Accepted", "difficulty": "Medium"},
            {"question_id": 4, "status": "Accepted", "difficulty": "Medium"},
            {"question_id": 3, "status": "Wrong Answer", "difficulty": "Medium"},
            {"question_id": 5, "status": "Accepted", "difficulty": "Hard"},
            {"question_id": 6, "status": "Wrong Answer", "difficulty": "Hard"},
        ])
        level = compute_user_difficulty_level(subs, self.qid_to_diff)
        self.assertEqual(level, 1)  # Medium

        # Now add 2 more Hard Accepted attempts (3/4 on Hard >= 50%)
        subs_hard = pd.concat([
            subs,
            pd.DataFrame([
                {"question_id": 5, "status": "Accepted", "difficulty": "Hard"},
                {"question_id": 6, "status": "Accepted", "difficulty": "Hard"},
            ])
        ], ignore_index=True)
        level_hard = compute_user_difficulty_level(subs_hard, self.qid_to_diff)
        self.assertEqual(level_hard, 2)  # Hard

    def test_popularity_rankings_and_counts(self):
        train_s = pd.DataFrame([
            {"user_id": 1, "question_id": 10, "status": "Accepted"},
            {"user_id": 2, "question_id": 10, "status": "Accepted"},
            {"user_id": 1, "question_id": 20, "status": "Accepted"},
            {"user_id": 3, "question_id": 20, "status": "Wrong Answer"},
            {"user_id": 1, "question_id": 30, "status": "Time Limit Exceeded"},
        ])
        all_qids = [10, 20, 30, 40]
        rankings = get_popularity_rankings(train_s, all_qids)
        self.assertEqual(rankings[0], 10)  # 2 unique users
        self.assertEqual(rankings[1], 20)  # 1 unique user

        counts = get_popularity_counts(train_s, all_qids)
        self.assertEqual(counts[10], 2)
        self.assertEqual(counts[20], 1)
        self.assertEqual(counts[30], 0)
        self.assertEqual(counts[40], 0)

    def test_hybrid_recommender_enforce_diversity_flag(self):
        # Create a toy questions dataframe where all items share tag 'Array'
        toy_q = pd.DataFrame([
            {"question_id": i, "title": f"P{i}", "difficulty": "Easy", "topic_tags": ["Array"], "acceptance_rate": 0.5}
            for i in range(1, 10)
        ])
        rec = HybridRecommender(toy_q)
        # With enforce_diversity=True and only 1 tag, diversity filter caps at 2, then falls back to fill
        res_div = rec.recommend(user_id=1, top_n=5, enforce_diversity=True)
        self.assertEqual(len(res_div), 5)

        # With enforce_diversity=False
        res_nodiv = rec.recommend(user_id=1, top_n=5, enforce_diversity=False)
        self.assertEqual(len(res_nodiv), 5)
        self.assertIn("recommendation_score", res_nodiv.columns)

    def test_benchmark_returns_8_methods_and_product_metrics(self):
        # Quick run on small user pool
        results = run_benchmark(
            seeds=[42],
            k_values=[5],
            generator_version="v1",
            n_users=50,
            n_submissions=500,
            verbose=False,
        )

        expected_methods = [
            "random",
            "popularity",
            "content_only",
            "als_only",
            "hybrid",
            "als_popularity_blend",
            "hybrid_popularity_blend",
            "hybrid_no_diversity",
        ]
        self.assertEqual(len(results["methods"]), 8)
        for m in expected_methods:
            self.assertIn(m, results["methods"])
            self.assertIn("precision@5", results["methods"][m])
            self.assertIn("weak_coverage@5", results["methods"][m])
            self.assertIn("difficulty_fit@5", results["methods"][m])

        expected_pairs = [
            "hybrid_minus_popularity",
            "hybrid_minus_random",
            "hybrid_minus_als_only",
        ]
        for p in expected_pairs:
            self.assertIn(p, results["paired_differences"])
            self.assertIn("precision@5", results["paired_differences"][p])
            self.assertIn("weak_coverage@5", results["paired_differences"][p])
            self.assertIn("difficulty_fit@5", results["paired_differences"][p])


if __name__ == "__main__":
    unittest.main()
