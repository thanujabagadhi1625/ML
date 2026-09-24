"""
tests/test_final_integrity.py
-----------------------------
Phase 6 integrity test suite verifying core benchmark, data, and catalogue guarantees:
  1. Same seed gives identical data (reproducibility).
  2. Timestamps strictly increase with sequence position (temporal integrity).
  3. No user appears in two splits (zero cross-split leakage).
  4. Catalogue contains no default-"Array" fallbacks (valid canonical mapping).
  5. Popularity distribution is long-tailed (Zipf heavy-tailed consumption).
  6. Benchmark script is strictly deterministic for a fixed seed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

import numpy as np
import pandas as pd

import config
from data_processing import (
    generate_synthetic_dataset_v3,
    load_canonical_questions,
    user_train_val_test_split,
)
from training.benchmark_recommenders import run_benchmark


class TestFinalIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions_df = load_canonical_questions()

    def test_1_same_seed_gives_identical_data(self):
        """Verify that identical random seeds produce byte-identical synthetic datasets."""
        u1, q1, s1, l1 = generate_synthetic_dataset_v3(
            self.questions_df, n_users=30, random_seed=42
        )
        u2, q2, s2, l2 = generate_synthetic_dataset_v3(
            self.questions_df, n_users=30, random_seed=42
        )

        pd.testing.assert_frame_equal(u1, u2)
        pd.testing.assert_frame_equal(q1, q2)
        pd.testing.assert_frame_equal(s1, s2)
        pd.testing.assert_frame_equal(l1, l2)

    def test_2_timestamps_increase_with_sequence(self):
        """Verify that event timestamps for every user strictly increase chronologically."""
        _, _, subs_df, _ = generate_synthetic_dataset_v3(
            self.questions_df, n_users=30, random_seed=42
        )
        for uid, grp in subs_df.groupby("user_id"):
            ts = grp["timestamp"].to_numpy()
            diffs = ts[1:] - ts[:-1]
            self.assertTrue(
                np.all(diffs > np.timedelta64(0, "ns")),
                f"User {uid} submissions are not strictly increasing chronologically!",
            )

    def test_3_no_user_appears_in_two_splits(self):
        """Verify zero user leakage across Train, Validation, and Test splits."""
        users_df, _, subs_df, _ = generate_synthetic_dataset_v3(
            self.questions_df, n_users=60, random_seed=42
        )
        train_u, val_u, test_u, train_s, val_s, test_s = user_train_val_test_split(
            users_df=users_df,
            submissions_df=subs_df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        train_uids = set(train_u["user_id"])
        val_uids = set(val_u["user_id"])
        test_uids = set(test_u["user_id"])

        self.assertEqual(len(train_uids & val_uids), 0, "Train and Val share users!")
        self.assertEqual(len(train_uids & test_uids), 0, "Train and Test share users!")
        self.assertEqual(len(val_uids & test_uids), 0, "Val and Test share users!")

        train_s_uids = set(train_s["user_id"])
        val_s_uids = set(val_s["user_id"])
        test_s_uids = set(test_s["user_id"])

        self.assertEqual(len(train_s_uids & val_s_uids), 0, "Train and Val share submission logs!")
        self.assertEqual(len(train_s_uids & test_s_uids), 0, "Train and Test share submission logs!")
        self.assertEqual(len(val_s_uids & test_s_uids), 0, "Val and Test share submission logs!")

        self.assertTrue(train_s_uids.issubset(train_uids))
        self.assertTrue(val_s_uids.issubset(val_uids))
        self.assertTrue(test_s_uids.issubset(test_uids))

    def test_4_catalogue_contains_no_default_array_fallbacks(self):
        """Verify that the canonical catalogue contains no default-'Array' unmapped fallbacks."""
        self.assertFalse(self.questions_df.empty, "Canonical catalogue is empty!")
        all_canonical_tags = set(config.TOPIC_TAGS)

        array_only_count = 0
        total_questions = len(self.questions_df)

        for _, row in self.questions_df.iterrows():
            tags = row["topic_tags"]
            self.assertIsInstance(tags, list, f"Question {row['question_id']} tags is not a list")
            self.assertGreater(len(tags), 0, f"Question {row['question_id']} has empty tags")
            for t in tags:
                self.assertIn(
                    t,
                    all_canonical_tags,
                    f"Tag '{t}' in question {row['question_id']} is not in config.TOPIC_TAGS!",
                )
            if tags == ["Array"]:
                array_only_count += 1

        array_only_ratio = array_only_count / float(total_questions)
        self.assertLess(
            array_only_ratio,
            0.25,
            f"Artificial 'Array' fallback detected: {array_only_ratio:.2%} questions are ['Array']",
        )

    def test_5_popularity_distribution_is_long_tailed(self):
        """Verify question popularity distribution follows a heavy-tailed power-law distribution."""
        _, _, subs_df, _ = generate_synthetic_dataset_v3(
            self.questions_df, n_users=80, random_seed=42, zipf_exponent=0.8
        )
        counts = subs_df["question_id"].value_counts()
        n_top = max(1, int(round(0.20 * len(counts))))
        top_share = counts.iloc[:n_top].sum() / counts.sum()

        self.assertGreater(
            top_share,
            0.45,
            f"Expected heavy-tailed popularity with top 20% share > 45%, got {top_share:.2%}",
        )

    def test_6_benchmark_script_is_deterministic_for_fixed_seed(self):
        """Verify that running the benchmark harness with a fixed seed produces identical output."""
        res1 = run_benchmark(
            seeds=[42],
            n_users=40,
            generator_version="v3",
            als_params={"n_factors": 16, "alpha": 5.0, "reg_lambda": 50.0, "n_epochs": 5},
            hybrid_weights=(1.0, 0.0, 0.0),
            verbose=False,
        )
        res2 = run_benchmark(
            seeds=[42],
            n_users=40,
            generator_version="v3",
            als_params={"n_factors": 16, "alpha": 5.0, "reg_lambda": 50.0, "n_epochs": 5},
            hybrid_weights=(1.0, 0.0, 0.0),
            verbose=False,
        )

        for m in ["random", "popularity", "als_only", "hybrid"]:
            for metric in ["precision@5", "recall@5", "hit_rate@5", "ndcg@5"]:
                v1 = res1["methods"][m][metric]["mean"]
                v2 = res2["methods"][m][metric]["mean"]
                self.assertAlmostEqual(
                    v1,
                    v2,
                    places=5,
                    msg=f"Non-deterministic benchmark output for method {m}, metric {metric}: {v1} != {v2}",
                )

    def test_7_catalogue_real_leetcode_ids(self):
        """Assert frontendQuestionId stored as leetcode_id for specified problems."""
        import json
        # Check loaded canonical DataFrame
        slug_to_lid = dict(zip(self.questions_df["slug"], self.questions_df["leetcode_id"]))
        self.assertEqual(slug_to_lid.get("two-sum"), 1)
        self.assertEqual(slug_to_lid.get("basic-calculator-ii"), 227)
        self.assertEqual(slug_to_lid.get("slowest-key"), 1629)
        self.assertEqual(slug_to_lid.get("beautiful-array"), 932)

        # Also check canonical_questions.json file directly
        items = json.loads(config.CANONICAL_QUESTIONS_PATH.read_text(encoding="utf-8"))
        json_slug_to_lid = {item["slug"]: item.get("leetcode_id") for item in items}
        self.assertEqual(json_slug_to_lid.get("two-sum"), 1)
        self.assertEqual(json_slug_to_lid.get("basic-calculator-ii"), 227)
        self.assertEqual(json_slug_to_lid.get("slowest-key"), 1629)
        self.assertEqual(json_slug_to_lid.get("beautiful-array"), 932)


if __name__ == "__main__":
    unittest.main()
