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


if __name__ == "__main__":
    unittest.main()
