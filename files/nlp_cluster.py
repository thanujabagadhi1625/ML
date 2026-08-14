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
        silhouette score. Gracefully handles low sample size or duplicate embeddings.
        """
        n_samples = embeddings.shape[0]
        if n_samples < 3:
            return 1

        unique_samples = np.unique(embeddings, axis=0).shape[0]
        if unique_samples < 2:
            return 1

        max_k = min(config.MAX_CLUSTERS, unique_samples, n_samples - 1)
        min_k = min(config.MIN_CLUSTERS, max_k)

        if max_k < 2 or min_k < 2:
            return 1

        best_k, best_score = 1, -1.0
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                if len(np.unique(labels)) < 2:
                    continue
                score = float(silhouette_score(embeddings, labels))
                if score > best_score:
                    best_k, best_score = k, score
            except Exception:
                continue
        return best_k

    # ---- Step 3: Clustering -----------------------------------------------------
    def cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int | None = None) -> np.ndarray:
        k = n_clusters or self._select_optimal_k(embeddings)
        if k <= 1:
            return np.zeros(embeddings.shape[0], dtype=int)
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        return kmeans.fit_predict(embeddings)

    # ---- Step 4: cluster labeling ------------------------------------------------
    @staticmethod
    def _label_clusters_with_tags(failed_submissions_df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, List[str]]:
        """
        Prefer actual problem tags over raw text keywords when we have them.
        Guarantees that every cluster ID has a non-empty key in the returned dictionary.
        """
        unique_clusters = sorted(list(set(cluster_labels)))
        labels: Dict[int, List[str]] = {}

        rows = failed_submissions_df[["topic_tags"]].copy()
        rows["cluster"] = cluster_labels
        rows = rows.explode("topic_tags").dropna(subset=["topic_tags"])
        rows["topic_tags"] = rows["topic_tags"].astype(str).str.strip()
        rows = rows[rows["topic_tags"] != ""]

        if not rows.empty:
            tag_counts = (
                rows.groupby(["cluster", "topic_tags"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            for c_id in unique_clusters:
                c_tags = (
                    tag_counts.loc[tag_counts["cluster"] == c_id]
                    .sort_values(["count", "topic_tags"], ascending=[False, True])
                    ["topic_tags"]
                    .head(config.TOP_KEYWORDS_PER_CLUSTER)
                    .tolist()
                )
                if c_tags:
                    labels[int(c_id)] = c_tags

        descriptions = failed_submissions_df["description"].fillna("").tolist()
        fallback_labels = WeakTopicAnalyzer._label_clusters_with_ctfidf(descriptions, cluster_labels)
        for c_id in unique_clusters:
            if int(c_id) not in labels or not labels[int(c_id)]:
                labels[int(c_id)] = fallback_labels.get(int(c_id), ["General"])

        return labels

    @staticmethod
    def _label_clusters_with_ctfidf(descriptions: List[str], cluster_labels: np.ndarray) -> Dict[int, List[str]]:
        """
        Fallback for exported data that does not provide meaningful topic tags.
        """
        df = pd.DataFrame({"text": descriptions, "cluster": cluster_labels})
        cluster_docs = df.groupby("cluster")["text"].apply(lambda texts: " ".join(texts))

        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_docs.values)
            vocab = np.array(vectorizer.get_feature_names_out())

            labels = {}
            for row_idx, cluster_id in enumerate(cluster_docs.index):
                row = tfidf_matrix[row_idx].toarray().ravel()
                if len(vocab) > 0:
                    top_indices = np.argsort(row)[::-1][: config.TOP_KEYWORDS_PER_CLUSTER]
                    labels[int(cluster_id)] = vocab[top_indices].tolist()
                else:
                    labels[int(cluster_id)] = ["General"]
            return labels
        except Exception:
            return {int(c): ["General"] for c in cluster_docs.index}

    # ---- Full pipeline for a single user -----------------------------------------
    def analyze_user_weak_topics(self, failed_submissions_df: pd.DataFrame) -> Dict:
        """
        Parameters
        ----------
        failed_submissions_df : DataFrame with at least a `description` column

        Returns
        -------
        dict with:
          - "clusters": list of {cluster_id, size, keywords, sample_titles}
          - "n_failed_submissions": int
        """
        n_failed = len(failed_submissions_df)
        if n_failed == 0:
            return {
                "clusters": [],
                "n_failed_submissions": 0,
                "note": "No failed submissions found. Excellent accuracy!",
            }

        def _structured_topic_fallback(note_str: str) -> Dict:
            if "topic_tags" not in failed_submissions_df.columns:
                sample_titles = failed_submissions_df["title"].head(3).tolist() if "title" in failed_submissions_df.columns else []
                return {
                    "clusters": [],
                    "n_failed_submissions": n_failed,
                    "note": note_str,
                }
            rows = failed_submissions_df[["topic_tags", "title"]].explode("topic_tags").dropna(subset=["topic_tags"])
            rows["topic_tags"] = rows["topic_tags"].astype(str).str.strip()
            rows = rows[rows["topic_tags"] != ""]
            if rows.empty:
                sample_titles = failed_submissions_df["title"].head(3).tolist() if "title" in failed_submissions_df.columns else []
                return {
                    "clusters": [{
                        "cluster_id": 0,
                        "size": n_failed,
                        "keywords": ["General"],
                        "sample_titles": sample_titles,
                    }],
                    "n_failed_submissions": n_failed,
                    "note": note_str,
                }
            top_tags = rows["topic_tags"].value_counts().head(5)
            clusters = []
            for idx, (tag, count) in enumerate(top_tags.items()):
                sample_t = rows[rows["topic_tags"] == tag]["title"].head(3).tolist()
                clusters.append({
                    "cluster_id": idx,
                    "size": int(count),
                    "keywords": [tag],
                    "sample_titles": sample_t,
                })
            return {
                "clusters": clusters,
                "n_failed_submissions": n_failed,
                "note": note_str,
            }

        if n_failed < config.MIN_FAILED_SUBMISSIONS_FOR_CLUSTERING:
            return _structured_topic_fallback("Fewer than 5 failed submissions; using structured topic analysis instead.")

        try:
            descriptions = failed_submissions_df["description"].tolist()
            embeddings = self.embed_descriptions(descriptions)
            cluster_labels = self.cluster_embeddings(embeddings)

            if len(set(cluster_labels)) < 2:
                return _structured_topic_fallback("NLP clustering yielded 1 cluster; using structured topic analysis instead.")

            keyword_map = self._label_clusters_with_tags(failed_submissions_df, cluster_labels)
            working = failed_submissions_df.assign(cluster=cluster_labels)
            clusters_summary = []
            for cluster_id, group in working.groupby("cluster"):
                clusters_summary.append({
                    "cluster_id": int(cluster_id),
                    "size": int(len(group)),
                    "keywords": keyword_map.get(int(cluster_id), ["General"]),
                    "sample_titles": group["title"].head(3).tolist(),
                })

            clusters_summary.sort(key=lambda c: c["size"], reverse=True)
            return {"clusters": clusters_summary, "n_failed_submissions": n_failed}
        except Exception as e:
            return _structured_topic_fallback(f"NLP clustering unavailable due to data variance ({e}); using structured topic analysis.")


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
