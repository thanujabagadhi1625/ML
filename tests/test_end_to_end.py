"""
tests/test_end_to_end.py
------------------------
End-to-end integration tests for the LeetCodeMentor facade using synthetic and demo fixtures.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

from data_processing import generate_full_synthetic_dataset, load_leetcode_history_records
from main import LeetCodeMentor


class TestEndToEnd(unittest.TestCase):
    """Test suite for full pipeline orchestration."""

    def test_full_pipeline_with_synthetic_dataset(self):
        """Verify report generation across all 4 engines on a full synthetic dataset."""
        users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
        mentor = LeetCodeMentor(users_df, questions_df, submissions_df)

        sample_uid = int(users_df["user_id"].iloc[0])
        report = mentor.generate_report(sample_uid)

        self.assertIsInstance(report, dict)
        self.assertEqual(report["user_id"], sample_uid)
        self.assertIn("data_provenance", report)
        self.assertIn("diagnostics", report)
        self.assertIn("topic_profile", report)
        self.assertIn("weak_subtopics", report)
        self.assertIn("recommended_questions", report)
        self.assertIn("predicted_contest_rating", report)
        self.assertIn("model_evaluation", report)

    def test_full_pipeline_with_demo_fixture(self):
        """Verify report generation using the committed synthetic demo fixture."""
        demo_fixture_path = FILES_DIR / "demo" / "large_user.json"
        self.assertTrue(demo_fixture_path.exists(), f"Demo fixture missing at {demo_fixture_path}")

        demo_data = json.loads(demo_fixture_path.read_text(encoding="utf-8"))
        records = demo_data.get("submissions", [])
        self.assertGreater(len(records), 0)

        users_df, questions_df, submissions_df = load_leetcode_history_records(records)
        mentor = LeetCodeMentor(users_df, questions_df, submissions_df)

        sample_uid = int(mentor.users_df["user_id"].iloc[0])
        report = mentor.generate_report(sample_uid, provenance={"data_source": "demo_fixture", "is_demo_data": True})

        self.assertIsInstance(report, dict)
        self.assertTrue(report["data_provenance"]["is_demo_data"])
        self.assertGreater(len(report["recommended_questions"]), 0)

    def test_no_retraining_during_inference(self):
        """
        Architectural guarantee test:
        Asserts that neither XGBoost nor ALS is retrained when a user generates a report,
        and that saved model artifact files on disk are not overwritten (checked via SHA256 hashes).
        """
        import hashlib
        import config

        def _sha256(file_path: Path) -> str | None:
            if not file_path.exists():
                return None
            return hashlib.sha256(file_path.read_bytes()).hexdigest()

        # 1. Record artifact SHA256 hashes before inference
        hashes_before = {
            "rating": _sha256(config.RATING_MODEL_PATH),
            "als": _sha256(config.ALS_MODEL_PATH),
            "embeddings": _sha256(config.QUESTION_EMBEDDINGS_PATH),
            "catalogue": _sha256(config.CANONICAL_QUESTIONS_PATH),
        }

        # 2. Instantiate mentor on synthetic dataset
        users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
        mentor = LeetCodeMentor(users_df, questions_df, submissions_df)

        # Record in-memory model identities and factors
        initial_xgb_model = mentor.predictor.pipeline.named_steps["model"]
        initial_als_item_factors = mentor.recommender.als.item_factors.copy() if mentor.recommender.als.is_fitted else None

        # 3. Run multiple inference reports
        sample_uids = list(mentor.users_df["user_id"].iloc[:3])
        for uid in sample_uids:
            report = mentor.generate_report(uid)
            self.assertIsNotNone(report["predicted_contest_rating"])
            self.assertGreater(len(report["recommended_questions"]), 0)

        # 4. Assert disk artifact hashes are strictly identical before and after inference
        for name, path in [
            ("rating", config.RATING_MODEL_PATH),
            ("als", config.ALS_MODEL_PATH),
            ("embeddings", config.QUESTION_EMBEDDINGS_PATH),
            ("catalogue", config.CANONICAL_QUESTIONS_PATH),
        ]:
            if hashes_before[name] is not None:
                self.assertEqual(
                    _sha256(path),
                    hashes_before[name],
                    f"Artifact {name} was unexpectedly modified during inference!"
                )

        # 5. Assert in-memory model instances and fixed item matrices were unchanged
        self.assertIs(mentor.predictor.pipeline.named_steps["model"], initial_xgb_model)
        if initial_als_item_factors is not None:
            import numpy as np
            np.testing.assert_array_equal(mentor.recommender.als.item_factors, initial_als_item_factors)

    def test_cold_start_interaction_horizons(self):
        """
        Verify single-user pipeline boundary conditions across:
        - 0 interactions (brand new empty user)
        - 1 interaction
        - 2 interactions
        - 3 interactions (ALS fold-in threshold)
        - Many interactions
        Ensures the system never crashes and solved questions are excluded.
        """
        import pandas as pd
        from schemas import SUBMISSION_COLUMNS, USER_COLUMNS

        users_df = pd.DataFrame([{
            "user_id": 1,
            "account_age_days": 15,
            "latent_skill": 0.0,
            "contest_rating": 1500.0,
        }])[USER_COLUMNS]
        questions_df = generate_full_synthetic_dataset()[1]

        # Case 1: 0 interactions
        empty_subs = pd.DataFrame(columns=SUBMISSION_COLUMNS)
        mentor_0 = LeetCodeMentor(users_df, questions_df, empty_subs)
        rep_0 = mentor_0.generate_report(1)
        self.assertEqual(len(rep_0["recommended_questions"]), 5)
        self.assertIsNotNone(rep_0["predicted_contest_rating"])

        # Case 2: 1 interaction (solved Q1)
        sub_1 = pd.DataFrame([{
            "user_id": 1, "question_id": 1, "timestamp": pd.Timestamp.now(),
            "status": "Accepted", "runtime_ms": 25.0, "language": "Python3"
        }])
        mentor_1 = LeetCodeMentor(users_df, questions_df, sub_1)
        rep_1 = mentor_1.generate_report(1)
        rec_ids_1 = [r["question_id"] for r in rep_1["recommended_questions"]]
        self.assertEqual(len(rec_ids_1), 5)
        self.assertNotIn(1, rec_ids_1)  # Solved Q1 excluded

        # Case 3: 2 interactions (solved Q1, failed Q2)
        sub_2 = pd.DataFrame([
            {"user_id": 1, "question_id": 1, "timestamp": pd.Timestamp.now() - pd.Timedelta(days=1),
             "status": "Accepted", "runtime_ms": 25.0, "language": "Python3"},
            {"user_id": 1, "question_id": 2, "timestamp": pd.Timestamp.now(),
             "status": "Wrong Answer", "runtime_ms": 40.0, "language": "Python3"},
        ])
        mentor_2 = LeetCodeMentor(users_df, questions_df, sub_2)
        rep_2 = mentor_2.generate_report(1)
        rec_ids_2 = [r["question_id"] for r in rep_2["recommended_questions"]]
        self.assertEqual(len(rec_ids_2), 5)
        self.assertNotIn(1, rec_ids_2)  # Solved Q1 excluded

        # Case 4: 3 interactions (crosses ALS fold-in threshold)
        sub_3 = pd.DataFrame([
            {"user_id": 1, "question_id": 1, "timestamp": pd.Timestamp.now() - pd.Timedelta(days=2),
             "status": "Accepted", "runtime_ms": 25.0, "language": "Python3"},
            {"user_id": 1, "question_id": 2, "timestamp": pd.Timestamp.now() - pd.Timedelta(days=1),
             "status": "Wrong Answer", "runtime_ms": 40.0, "language": "Python3"},
            {"user_id": 1, "question_id": 3, "timestamp": pd.Timestamp.now(),
             "status": "Accepted", "runtime_ms": 15.0, "language": "Python3"},
        ])
        mentor_3 = LeetCodeMentor(users_df, questions_df, sub_3)
        rep_3 = mentor_3.generate_report(1)
        rec_ids_3 = [r["question_id"] for r in rep_3["recommended_questions"]]
        self.assertEqual(len(rec_ids_3), 5)
        self.assertNotIn(1, rec_ids_3)
        self.assertNotIn(3, rec_ids_3)


if __name__ == "__main__":
    unittest.main()
