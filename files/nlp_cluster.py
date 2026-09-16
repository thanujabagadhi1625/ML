"""
nlp_cluster.py
--------------
WEAK TOPICS ENGINE

Pipeline:
  1. Ingest a user's FAILED submissions (Wrong Answer / TLE / Runtime / Compile Error).
  2. Embed each problem's description with sentence-transformers (all-MiniLM-L6-v2).
  3. Cluster the embeddings with K-Means, auto-selecting k via silhouette score.
  4. Label each cluster with representative keywords using canonical class-based TF-IDF
     (c-TF-IDF, Grootendorst 2022) so clusters read as human-interpretable "weak sub-topics"
     rather than opaque cluster IDs.

The topic groupings emerge from embedding geometry, and cluster labels provide
human-interpretable semantic keywords.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

import config


# ==============================================================================
# Canonical Class-based TF-IDF (c-TF-IDF)
# ==============================================================================

def compute_ctfidf(
    documents_per_cluster: Dict[int, List[str]],
    top_n: int = config.TOP_KEYWORDS_PER_CLUSTER,
) -> Dict[int, List[str]]:
    """
    Canonical class-based TF-IDF (c-TF-IDF) formulation (Grootendorst, 2022).

    Given K clusters, we treat each cluster c as a distinct class by concatenating
    all problem descriptions in cluster c into a single class document.

    Mathematical Formulation:
        W_{t, c} = (tf_{t, c} / w_c) * ln(1 + A / tf_t)
    where:
        - tf_{t, c} is the frequency of term t in class/cluster c
        - w_c = sum_t tf_{t, c} is the total number of words in class/cluster c
        - A = (1 / K) * sum_c w_c is the average number of words per class
        - tf_t = sum_c tf_{t, c} is the total frequency of term t across all classes

    For each class c, terms with highest W_{t, c} are selected as representative keywords.
    """
    clusters = sorted(list(documents_per_cluster.keys()))
    if not clusters:
        return {}

    class_docs = []
    for c in clusters:
        texts = [str(t).strip() for t in documents_per_cluster[c] if str(t).strip()]
        class_docs.append(" ".join(texts) if texts else "general problem")

    vectorizer = CountVectorizer(stop_words="english", max_features=500, min_df=1)
    try:
        count_matrix = vectorizer.fit_transform(class_docs)
    except ValueError:
        # Vocabulary empty (e.g. empty inputs or only English stop words)
        return {c: ["General"] for c in clusters}

    vocab = np.array(vectorizer.get_feature_names_out())
    if len(vocab) == 0:
        return {c: ["General"] for c in clusters}

    # C is (K, V) raw count matrix
    C = count_matrix.toarray().astype(float)
    K, V = C.shape

    # w_c: total words per class (shape: K, 1)
    w_c = C.sum(axis=1, keepdims=True)
    # Average words per class across all clusters
    A = float(w_c.mean()) if K > 0 else 1.0

    # tf_t: total frequency of word across all classes (shape: 1, V)
    tf_t = C.sum(axis=0, keepdims=True)

    # Class-based term frequency: tf_{t, c} / w_c
    tf_norm = np.divide(C, np.maximum(w_c, 1e-9))

    # Inverse class frequency: ln(1 + A / tf_t)
    icf = np.log(1.0 + np.divide(A, np.maximum(tf_t, 1e-9)))

    # Canonical c-TF-IDF matrix (K, V)
    W = tf_norm * icf

    cluster_keywords: Dict[int, List[str]] = {}
    for row_idx, c_id in enumerate(clusters):
        row_scores = W[row_idx]
        if np.all(row_scores <= 0.0):
            cluster_keywords[c_id] = ["General"]
            continue
        top_indices = np.argsort(row_scores)[::-1][:top_n]
        valid_indices = [idx for idx in top_indices if row_scores[idx] > 0]
        if valid_indices:
            cluster_keywords[c_id] = vocab[valid_indices].tolist()
        else:
            cluster_keywords[c_id] = ["General"]

    return cluster_keywords


class WeakTopicAnalyzer:
    """
    Encapsulates the sentence-transformers embedding model, K-Means clustering,
    and canonical c-TF-IDF keyword extraction with robust edge-case degradation.
    """

    def __init__(self, model_name: str = config.SENTENCE_TRANSFORMER_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compute_ctfidf(self, texts: List[str], labels: List[int], top_n: int = 6) -> Dict[int, List[str]]:
        """Convenience wrapper around canonical c-TF-IDF calculation."""
        docs_per_cluster: Dict[int, List[str]] = {}
        for text, label in zip(texts, labels):
            docs_per_cluster.setdefault(label, []).append(text)
        return compute_ctfidf(docs_per_cluster, top_n=top_n)

    # ---- Step 1: Embedding ----------------------------------------------------
    def embed_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """Batch-encodes descriptions into dense embeddings with unit norm."""
        cleaned = [d.strip() if d and d.strip() else "algorithmic problem statement" for d in descriptions]
        embeddings = self.model.encode(
            cleaned,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings)

    # ---- Step 2: Optimal k selection -------------------------------------------
    def _select_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Sweeps k in [MIN_CLUSTERS, MAX_CLUSTERS] maximizing silhouette score.
        Gracefully handles small sample size, duplicate embeddings, or single-region clusters.
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
        """Clusters failure embeddings into optimal k semantic groups."""
        k = n_clusters or self._select_optimal_k(embeddings)
        if k <= 1:
            return np.zeros(embeddings.shape[0], dtype=int)
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        return kmeans.fit_predict(embeddings)

    # ---- Step 4: Cluster Labeling via Canonical c-TF-IDF & Topic Tags -----------
    @staticmethod
    def _label_clusters_with_tags(failed_submissions_df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, List[str]]:
        """
        Extracts human-interpretable keywords for each cluster using canonical c-TF-IDF,
        supplemented with empirical topic tags from problem metadata.
        """
        unique_clusters = sorted(list(set(cluster_labels)))

        # Group descriptions by cluster for c-TF-IDF
        descriptions_per_cluster: Dict[int, List[str]] = {int(c): [] for c in unique_clusters}
        for desc, c_id in zip(failed_submissions_df["description"].fillna(""), cluster_labels):
            descriptions_per_cluster[int(c_id)].append(str(desc))

        ctfidf_keywords = compute_ctfidf(descriptions_per_cluster, top_n=config.TOP_KEYWORDS_PER_CLUSTER)

        # Count empirical problem tags in cluster
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

        # Combine topic tags and canonical c-TF-IDF keywords
        final_labels: Dict[int, List[str]] = {}
        for c_id in unique_clusters:
            c_int = int(c_id)
            tag_list = labels.get(c_int, [])
            ctfidf_list = ctfidf_keywords.get(c_int, [])

            combined = []
            for item in tag_list + ctfidf_list:
                item_clean = item.strip().title()
                if item_clean and item_clean not in combined:
                    combined.append(item_clean)

            final_labels[c_int] = combined[:config.TOP_KEYWORDS_PER_CLUSTER] if combined else ["General"]

        return final_labels

    # ---- Full Pipeline for a Single User ----------------------------------------
    def analyze_user_weak_topics(self, failed_submissions_df: pd.DataFrame) -> Dict:
        """
        Main entry point for single-user failure analysis.
        Handles zero-failure, low-failure, duplicate, and single-cluster edge cases gracefully.
        """
        n_failed = len(failed_submissions_df)
        if n_failed == 0:
            return {
                "clusters": [],
                "n_failed_submissions": 0,
                "note": "No failed submissions found. Excellent accuracy!",
            }

        def _structured_fallback(note_str: str) -> Dict:
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

        # Guard against insufficient samples for meaningful embedding clustering
        if n_failed < config.MIN_FAILED_SUBMISSIONS_FOR_CLUSTERING:
            return _structured_fallback(
                f"Fewer than {config.MIN_FAILED_SUBMISSIONS_FOR_CLUSTERING} failed submissions; using structured topic analysis instead."
            )

        try:
            descriptions = failed_submissions_df["description"].fillna("").tolist()
            embeddings = self.embed_descriptions(descriptions)
            cluster_labels = self.cluster_embeddings(embeddings)

            # If all failures fall into a single semantic region
            if len(set(cluster_labels)) < 2:
                keyword_map = self._label_clusters_with_tags(failed_submissions_df, cluster_labels)
                sample_titles = failed_submissions_df["title"].head(3).tolist() if "title" in failed_submissions_df.columns else []
                return {
                    "clusters": [{
                        "cluster_id": 0,
                        "size": n_failed,
                        "keywords": keyword_map.get(0, ["General"]),
                        "sample_titles": sample_titles,
                    }],
                    "n_failed_submissions": n_failed,
                    "note": "All failure embeddings belong to a single coherent semantic cluster.",
                }

            keyword_map = self._label_clusters_with_tags(failed_submissions_df, cluster_labels)
            working = failed_submissions_df.assign(cluster=cluster_labels)
            clusters_summary = []
            for cluster_id, group in working.groupby("cluster"):
                clusters_summary.append({
                    "cluster_id": int(cluster_id),
                    "size": int(len(group)),
                    "keywords": keyword_map.get(int(cluster_id), ["General"]),
                    "sample_titles": group["title"].head(3).tolist() if "title" in group.columns else [],
                })

            clusters_summary.sort(key=lambda c: c["size"], reverse=True)
            return {"clusters": clusters_summary, "n_failed_submissions": n_failed}
        except Exception as e:
            return _structured_fallback(f"NLP clustering degraded gracefully ({e}); using structured topic analysis.")


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
