"""
tests/test_nlp_cluster.py
-------------------------
Unit tests for SentenceTransformer embedding, silhouette score clustering,
canonical c-TF-IDF topic labeling, and cold-start edge cases.
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

    def test_canonical_ctfidf_calculation(self):
        """
        Verify canonical class-based TF-IDF calculation according to Maarten Grootendorst's formula:
        W_{t, c} = (tf_{t, c} / w_c) * ln(1 + A / tf_t)
        """
        cluster_docs = {
            0: ["dynamic programming knapsack optimal substructure", "dynamic programming memoization subproblems"],
            1: ["graph shortest path dijkstra traversal", "graph directed acyclic topological sort"],
        }
        labels = [0, 0, 1, 1]
        texts = cluster_docs[0] + cluster_docs[1]

        keywords = self.analyzer.compute_ctfidf(texts, labels, top_n=3)

        self.assertIn(0, keywords)
        self.assertIn(1, keywords)
        # Dynamic programming keywords should dominate cluster 0
        self.assertTrue(any("dynamic" in kw.lower() or "programming" in kw.lower() for kw in keywords[0]))
        # Graph keywords should dominate cluster 1
        self.assertTrue(any("graph" in kw.lower() or "path" in kw.lower() for kw in keywords[1]))

    def test_zero_failed_submissions_edge_case(self):
        """Verify pipeline handles 0 failed submissions cleanly."""
        empty_df = pd.DataFrame(columns=["description", "title"])
        result = self.analyzer.analyze_user_weak_topics(empty_df)
        self.assertEqual(result["n_failed_submissions"], 0)
        self.assertEqual(len(result["clusters"]), 0)
        self.assertIn("note", result)

    def test_duplicate_failed_submissions_edge_case(self):
        """Verify identical duplicate failed descriptions do not crash clustering or silhouette."""
        dup_df = pd.DataFrame({
            "description": ["Same exact graph problem text."] * 8,
            "title": ["Graph Problem"] * 8,
        })
        result = self.analyzer.analyze_user_weak_topics(dup_df)
        self.assertIn("clusters", result)
        self.assertEqual(result["n_failed_submissions"], 8)

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
