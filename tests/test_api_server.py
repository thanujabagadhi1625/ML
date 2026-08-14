"""
tests/test_api_server.py
------------------------
Integration tests for FastAPI sync and status endpoints using synthetic payload structures.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

from api_server import SyncPayload, SyncSubmission, _merge_sync_payload, _load_sync_store, _save_sync_store


class TestAPIServer(unittest.TestCase):
    """Test suite for local FastAPI submission sync logic."""

    def setUp(self):
        # Create a mock synthetic submission payload
        self.sample_submissions = [
            SyncSubmission(
                submission_id="test_sub_1",
                problem_slug="two-sum",
                problem_title="Two Sum",
                difficulty="Easy",
                status="Accepted",
                language="python3",
                user_id=1,
                timestamp="1780000000",
                runtime=15.0,
                memory=16.0,
                topics=["Array", "Hash Table"],
            ),
            SyncSubmission(
                submission_id="test_sub_2",
                problem_slug="course-schedule",
                problem_title="Course Schedule",
                difficulty="Medium",
                status="Wrong Answer",
                language="python3",
                user_id=1,
                timestamp="1780000100",
                runtime=None,
                memory=None,
                topics=["Graph", "Depth-First Search", "Breadth-First Search"],
            ),
        ]

    def test_sync_payload_validation(self):
        """Verify SyncPayload and SyncSubmission Pydantic validation."""
        payload = SyncPayload(
            username="synthetic_test_user",
            submissions=self.sample_submissions,
            sync_mode="sync_new",
            rebuild_history=False,
            fetch_limit=100,
        )
        self.assertEqual(payload.username, "synthetic_test_user")
        self.assertEqual(len(payload.submissions), 2)
        self.assertEqual(payload.submissions[0].difficulty, "Easy")

    def test_idempotent_sync_merge(self):
        """Verify incremental sync does not duplicate identical submission records."""
        payload = SyncPayload(
            username="synthetic_test_user",
            submissions=self.sample_submissions,
            sync_mode="sync_new",
            rebuild_history=False,
        )

        merged_first = _merge_sync_payload(payload)
        self.assertGreaterEqual(merged_first["total_submissions"], 2)

        # Syncing exact same payload a second time should add 0 new items
        merged_second = _merge_sync_payload(payload)
        self.assertEqual(merged_second["new_submissions"], 0)

        # Clean up test user from store to leave environment clean
        store = _load_sync_store()
        if "synthetic_test_user" in store:
            del store["synthetic_test_user"]
            _save_sync_store(store)


if __name__ == "__main__":
    unittest.main()
