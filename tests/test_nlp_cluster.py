"""
tests/test_nlp_cluster.py
-------------------------
Unit tests for SentenceTransformer embedding, silhouette score clustering, and c-TF-IDF topic labeling.
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
from nlp_cluster import WeakTopicAnalyzer


class TestNLPCluster(unittest.TestCase):
    """Test suite for the Weak Topic NLP clustering engine."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = WeakTopicAnalyzer()
        cls.users_df, cls.questions_df, cls.submissions_df = generate_full_synthetic_dataset()
        cls.fe = FeatureEngineer(cls.users_df, cls.questions_df, cls.submissions_df)

    def test_embedding_shape_and_norm(self):
        """Verify embeddings are batch-encoded with unit norm."""
        sample_texts = [
            "Given an array of integers, find the maximum subarray sum using Dynamic Programming.",
            "Determine whether a valid path exists in a weighted graph using Breadth-First Search.",
            "Find the lowest common ancestor in a Binary Search Tree.",
        ]
        embeddings = self.analyzer.embed_descriptions(sample_texts)
        self.assertEqual(embeddings.shape, (3, 384))

        # Check unit norm (cosine similarity friendly)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), rtol=1e-5)

    def test_insufficient_failed_submissions_guard(self):
        """Verify cold-start guard triggers when failed submission count is small."""
        small_failed_df = pd.DataFrame({
            "description": ["Single failed problem"],
            "title": ["Problem 1"],
        })
        result = self.analyzer.analyze_user_weak_topics(small_failed_df)
        self.assertEqual(result["n_failed_submissions"], 1)
        self.assertEqual(len(result["clusters"]), 0)
        self.assertIn("note", result)

    def test_full_weak_topics_pipeline(self):
        """Verify full clustering and c-TF-IDF keyword extraction."""
        failed = self.fe.get_failed_submissions_with_text()
        sample_failed = failed.head(20)

        result = self.analyzer.analyze_user_weak_topics(sample_failed)
        self.assertIn("clusters", result)
        self.assertIn("n_failed_submissions", result)

        if result["clusters"]:
            first_cluster = result["clusters"][0]
            self.assertIn("cluster_id", first_cluster)
            self.assertIn("size", first_cluster)
            self.assertIn("keywords", first_cluster)
            self.assertIn("sample_titles", first_cluster)


if __name__ == "__main__":
    unittest.main()
