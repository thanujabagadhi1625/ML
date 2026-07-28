"""
nlp_cluster.py
--------------
WEAK TOPICS ENGINE

Pipeline:
  1. Take a user's FAILED submissions (Wrong Answer / TLE / Runtime / Compile Error).
  2. Embed each problem's description with sentence-transformers (all-MiniLM-L6-v2).
  3. Cluster the embeddings with K-Means, auto-selecting k via silhouette score.
  4. Label each cluster with representative keywords using class-based TF-IDF
     (c-TF-IDF) so clusters read as human-interpretable "weak sub-topics"
     rather than opaque cluster IDs -- e.g. "graph traversal + backtracking"
     instead of "Cluster 3".

This deliberately avoids hand-written keyword-matching / if-else topic
classification: the topic groupings emerge purely from embedding geometry,
so it generalizes to problem phrasings never seen before.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import config


class WeakTopicAnalyzer:
    """
    Encapsulates the embedding model + clustering logic so it is loaded once
    and reused across users (loading a transformer per call would dominate
    runtime).
    """

    def __init__(self, model_name: str = config.SENTENCE_TRANSFORMER_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        # Lazy-load: the transformer is only pulled into memory the first time
        # it's actually needed, keeping cold-start / unit-test time low.
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ---- Step 1: Embedding ----------------------------------------------------
    def embed_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """Batch-encodes descriptions into dense embeddings. Fully vectorized
        (single batched forward pass), never one string at a time."""
        embeddings = self.model.encode(
            descriptions,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit-norm -> cosine similarity == dot product
        )
        return np.asarray(embeddings)

    # ---- Step 2: Optimal k selection -------------------------------------------
    def _select_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Sweeps k in [MIN_CLUSTERS, MAX_CLUSTERS] and picks the k maximizing
        silhouette score. This is model-selection, not a hand-tuned heuristic --
        the number of "weak sub-topics" a user has is not knowable in advance.
        """
        n_samples = embeddings.shape[0]
        max_k = min(config.MAX_CLUSTERS, n_samples - 1)
        min_k = min(config.MIN_CLUSTERS, max_k)

        if max_k < 2:
            return 1

        best_k, best_score = min_k, -1.0
        for k in range(min_k, max_k + 1):
            labels = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10).fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    # ---- Step 3: Clustering -----------------------------------------------------
    def cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int | None = None) -> np.ndarray:
        k = n_clusters or self._select_optimal_k(embeddings)
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        return kmeans.fit_predict(embeddings)

    # ---- Step 4: c-TF-IDF cluster labeling ---------------------------------------
    @staticmethod
    def _label_clusters_with_ctfidf(descriptions: List[str], cluster_labels: np.ndarray) -> Dict[int, List[str]]:
        """
        Class-based TF-IDF: concatenate all documents belonging to a cluster
        into one "mega-document" per cluster, then run TF-IDF treating each
        cluster as a single class. Terms that are frequent within a cluster
        but rare across other clusters bubble to the top -- giving each
        cluster a short, human-readable keyword signature.
        """
        df = pd.DataFrame({"text": descriptions, "cluster": cluster_labels})
        cluster_docs = df.groupby("cluster")["text"].apply(lambda texts: " ".join(texts))

        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = vectorizer.fit_transform(cluster_docs.values)
        vocab = np.array(vectorizer.get_feature_names_out())

        labels = {}
        for row_idx, cluster_id in enumerate(cluster_docs.index):
            row = tfidf_matrix[row_idx].toarray().ravel()
            top_indices = np.argsort(row)[::-1][: config.TOP_KEYWORDS_PER_CLUSTER]
            labels[int(cluster_id)] = vocab[top_indices].tolist()
        return labels

    # ---- Full pipeline for a single user -----------------------------------------
    def analyze_user_weak_topics(self, failed_submissions_df: pd.DataFrame) -> Dict:
        """
        Parameters
        ----------
        failed_submissions_df : DataFrame with at least a `description` column
            (typically produced by data_processing.FeatureEngineer.get_failed_submissions_with_text)

        Returns
        -------
        dict with:
          - "clusters": list of {cluster_id, size, keywords, sample_titles}
          - "n_failed_submissions": int
        """
        n_failed = len(failed_submissions_df)
        if n_failed < config.MIN_FAILED_SUBMISSIONS_FOR_CLUSTERING:
            return {"clusters": [], "n_failed_submissions": n_failed,
                    "note": "Not enough failed submissions yet for reliable clustering."}

        descriptions = failed_submissions_df["description"].tolist()
        embeddings = self.embed_descriptions(descriptions)
        cluster_labels = self.cluster_embeddings(embeddings)
        keyword_map = self._label_clusters_with_ctfidf(descriptions, cluster_labels)

        working = failed_submissions_df.assign(cluster=cluster_labels)
        clusters_summary = []
        for cluster_id, group in working.groupby("cluster"):
            clusters_summary.append({
                "cluster_id": int(cluster_id),
                "size": int(len(group)),
                "keywords": keyword_map[int(cluster_id)],
                "sample_titles": group["title"].head(3).tolist(),
            })

        clusters_summary.sort(key=lambda c: c["size"], reverse=True)
        return {"clusters": clusters_summary, "n_failed_submissions": n_failed}


if __name__ == "__main__":
    from data_processing import generate_full_synthetic_dataset, FeatureEngineer

    users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
    fe = FeatureEngineer(users_df, questions_df, submissions_df)

    sample_user_id = int(users_df["user_id"].iloc[0])
    failed = fe.get_failed_submissions_with_text(user_id=sample_user_id)

    analyzer = WeakTopicAnalyzer()
    result = analyzer.analyze_user_weak_topics(failed)

    print(f"User {sample_user_id} -- {result['n_failed_submissions']} failed submissions")
    for c in result.get("clusters", []):
        print(f"  Cluster {c['cluster_id']} (size={c['size']}): keywords={c['keywords']}")
