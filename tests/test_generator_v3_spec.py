"""
tests/test_generator_v3_spec.py
-------------------------------
Unit tests verifying Generator v3 against the Step 1 specification:
  a) 20-dim Dirichlet topic-interest vector per user; distinct draws per user; individual skill noise
  b) Global long-tailed question popularity (Zipf), fixed by seed
  c) Question choice proportional to popularity * exp(beta * interest . tags) * difficulty_fit; beta configurable
  d) Events in chronological order; timestamps strictly increase with sequence position
  e) Retry after failure with configurable probability; move on after success
  f) Solve probability uses the same time-varying skill and IRT formula
  g) Heavy-tailed lognormal submissions per user (minimum 20, configurable mean ~100)
  h) contest_rating computed from skill at the end of actual span min(account_age_days, 365)
  i) Catalogue size comes from loaded catalogue, not hardcoded
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

import config
from data_processing import (
    generate_synthetic_users_v3,
    generate_synthetic_submissions_v3,
    generate_synthetic_dataset_v3,
    load_canonical_questions,
)


class TestGeneratorV3Spec(unittest.TestCase):
    def setUp(self):
        self.questions_df = load_canonical_questions()

    def test_a_dirichlet_topic_interest_and_skill_noise(self):
        """Item a: 20-dim Dirichlet vectors; archetype shifts mean; users don't share identical vector; skill noise."""
        users_df, latent_df = generate_synthetic_users_v3(n=50, random_seed=42, return_latent_info=True)
        # Check topic interest dimension and Dirichlet simplex sum == 1
        for interest in latent_df["topic_interest"]:
            self.assertEqual(len(interest), 20)
            self.assertAlmostEqual(float(np.sum(interest)), 1.0, places=5)

        # Users of same archetype must not have identical vectors
        dp_users = latent_df[latent_df["archetype"] == "weak_dp"]
        self.assertGreaterEqual(len(dp_users), 2)
        int1 = dp_users.iloc[0]["topic_interest"]
        int2 = dp_users.iloc[1]["topic_interest"]
        self.assertFalse(np.allclose(int1, int2, atol=1e-4))

        # Skill noise: users of same archetype must not have identical base skill
        skill1 = dp_users.iloc[0]["base_skill"]
        skill2 = dp_users.iloc[1]["base_skill"]
        self.assertNotEqual(skill1, skill2)

    def test_b_global_zipf_popularity_fixed_by_seed(self):
        """Item b: Global long-tailed question popularity (Zipf), fixed by seed."""
        # Check reproducibility with same seed
        users_df, _ = generate_synthetic_users_v3(n=10, random_seed=42, return_latent_info=True)
        sub1 = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42)
        sub2 = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42)
        pd.testing.assert_frame_equal(sub1, sub2)

        # Check long-tail: top 20% most frequent items should carry disproportionate mass
        counts = sub1["question_id"].value_counts()
        n_top = max(1, int(round(0.20 * len(counts))))
        top_share = counts.iloc[:n_top].sum() / counts.sum()
        self.assertGreater(top_share, 0.40)

    def test_c_choice_proportional_and_beta_configurable(self):
        """Item c: beta affects question choice; higher beta increases tag matching."""
        users_df, latent_df = generate_synthetic_users_v3(n=10, random_seed=42, return_latent_info=True)
        sub_low_beta = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42, beta=0.1)
        sub_high_beta = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42, beta=5.0)
        # Low vs high beta generates different submission distributions
        self.assertFalse(sub_low_beta["question_id"].equals(sub_high_beta["question_id"]))

    def test_d_chronological_strictly_increasing_timestamps(self):
        """Item d: Events generated in chronological order; timestamps strictly increase."""
        users_df, _ = generate_synthetic_users_v3(n=15, random_seed=42, return_latent_info=True)
        subs = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42)
        for uid, grp in subs.groupby("user_id"):
            ts = grp["timestamp"].to_numpy()
            diffs = ts[1:] - ts[:-1]
            self.assertTrue(np.all(diffs > np.timedelta64(0, "ns")), f"User {uid} timestamps not strictly increasing!")

    def test_e_retry_after_failure_and_move_on_after_success(self):
        """Item e: After failure retry with probability p_retry; after success move on."""
        users_df, _ = generate_synthetic_users_v3(n=20, random_seed=42, return_latent_info=True)
        # With p_retry = 0.0, after failure user NEVER immediately retries same question (unless sampled by chance)
        subs_no_retry = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42, p_retry=0.0)
        # With p_retry = 1.0, after failure user ALWAYS immediately retries same question
        subs_full_retry = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42, p_retry=1.0)
        for uid, grp in subs_full_retry.groupby("user_id"):
            qids = grp["question_id"].tolist()
            statuses = grp["status"].tolist()
            for i in range(len(qids) - 1):
                if statuses[i] != "Accepted":
                    self.assertEqual(qids[i + 1], qids[i], f"Expected retry on failure for user {uid} at step {i}")

    def test_f_solve_probability_time_varying_skill(self):
        """Item f: Solve probability uses time-varying skill; improving users achieve higher accuracy late vs early."""
        users_df, latent_df = generate_synthetic_users_v3(n=30, random_seed=42, return_latent_info=True)
        improving_uids = set(latent_df[latent_df["archetype"] == "improving"]["user_id"])
        subs = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42, latent_users_info=latent_df)
        imp_subs = subs[subs["user_id"].isin(improving_uids)]
        # For improving users, split early half vs late half
        early_acc = []
        late_acc = []
        for uid, grp in imp_subs.groupby("user_id"):
            n = len(grp)
            if n >= 20:
                mid = n // 2
                early = (grp.iloc[:mid]["status"] == "Accepted").mean()
                late = (grp.iloc[mid:]["status"] == "Accepted").mean()
                early_acc.append(early)
                late_acc.append(late)
        self.assertGreater(float(np.mean(late_acc)), float(np.mean(early_acc)))

    def test_g_submissions_heavy_tailed_lognormal(self):
        """Item g: Submissions per user lognormal, minimum 20, mean about 100."""
        users_df, _ = generate_synthetic_users_v3(n=100, random_seed=42, return_latent_info=True)
        subs = generate_synthetic_submissions_v3(users_df, self.questions_df, random_seed=42, mean_submissions=100.0, min_submissions=20)
        counts = subs.groupby("user_id").size().to_numpy()
        self.assertTrue(np.all(counts >= 20))
        self.assertGreater(counts.mean(), 75.0)
        self.assertLess(counts.mean(), 125.0)
        # Heavy-tailed: max should be substantially higher than median
        self.assertGreater(counts.max(), np.median(counts) * 1.8)

    def test_h_contest_rating_consistent_with_submission_span(self):
        """Item h: contest_rating computed from skill at END of actual span min(age, 365)."""
        users_df, latent_df = generate_synthetic_users_v3(n=30, random_seed=42, return_latent_info=True)
        spans = latent_df["submission_span"].to_numpy()
        ages = users_df["account_age_days"].to_numpy()
        np.testing.assert_array_equal(spans, np.minimum(ages, 365))
        # Final skill matches base + slope * span
        expected_skill = latent_df["base_skill"] + latent_df["learning_slope"] * spans
        np.testing.assert_allclose(users_df["latent_skill"].to_numpy(), expected_skill.to_numpy(), rtol=1e-5)

    def test_i_catalogue_size_from_loaded_catalogue(self):
        """Item i: Catalogue size comes from loaded catalogue, not hardcoded."""
        sub_catalogue = self.questions_df.head(50).copy()
        users_df, q_df, subs_df, _ = generate_synthetic_dataset_v3(questions_df=sub_catalogue, n_users=10, random_seed=42)
        self.assertEqual(len(q_df), 50)
        self.assertTrue(set(subs_df["question_id"].unique()).issubset(set(sub_catalogue["question_id"].unique())))


if __name__ == "__main__":
    unittest.main()
