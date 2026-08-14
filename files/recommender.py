"""
recommender.py
--------------
RECOMMENDATION ENGINE

Two complementary signals, blended into a hybrid score:

  1. Collaborative Filtering via implicit-feedback Matrix Factorization,
     trained with Alternating Least Squares (ALS) on the user-item
     interaction matrix (weighted by solve status). This captures
     "users like you tend to solve/attempt these next" patterns.

  2. Content-based similarity using TF-IDF over topic tags + difficulty,
     which solves the cold-start problem for brand-new users/questions
     that ALS has no interaction history for.

Both stages are expressed as matrix operations (sparse matrices + numpy
linalg), not per-pair loops, so this scales to the full question bank.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


# ==============================================================================
# 1. Implicit-Feedback ALS Matrix Factorization
# ==============================================================================

class ALSMatrixFactorization:
    """
    Alternating Least Squares for implicit feedback (Hu, Koren & Volinsky, 2008).

    Given a binary interaction matrix R (user attempted/solved item),
    confidence C = 1 + alpha * R, we alternately solve closed-form
    least-squares systems for user factors X and item factors Y:

        x_u = (Y^T Cu Y + lambda*I)^-1 Y^T Cu p_u
        y_i = (X^T Ci X + lambda*I)^-1 X^T Ci p_i

    Each solve is a vectorized linear system (np.linalg.solve), which is why
    ALS scales far better than naive SGD-per-interaction for implicit data.
    """

    def __init__(
        self,
        n_factors: int = config.MF_LATENT_DIM,
        reg_lambda: float = config.MF_REG_LAMBDA,
        alpha: float = config.MF_CONFIDENCE_ALPHA,
        n_epochs: int = config.MF_EPOCHS,
        random_state: int = config.RANDOM_SEED,
    ):
        self.n_factors = n_factors
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(random_state)

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_index_: dict | None = None
        self.item_index_: dict | None = None

    def fit(self, interactions: pd.DataFrame, user_col="user_id", item_col="question_id", weight_col="weight"):
        """
        Parameters
        ----------
        interactions : DataFrame with columns [user_col, item_col, weight_col]
            weight_col encodes interaction strength, e.g. 1.0 for attempted,
            2.0 for solved -- see build_interaction_matrix() below.
        """
        users = interactions[user_col].unique()
        items = interactions[item_col].unique()
        self.user_index_ = {u: i for i, u in enumerate(users)}
        self.item_index_ = {it: i for i, it in enumerate(items)}

        rows = interactions[user_col].map(self.user_index_).to_numpy()
        cols = interactions[item_col].map(self.item_index_).to_numpy()
        vals = interactions[weight_col].to_numpy()

        n_users, n_items = len(users), len(items)
        R = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        # Confidence is only ever needed at OBSERVED (nonzero) entries -- the implicit
        # "+1" baseline confidence for unobserved entries is handled algebraically via
        # the YtY / XtX terms below, so we keep C sparse rather than densifying it.
        C = R.copy()
        C.data = 1.0 + self.alpha * C.data

        self.user_factors = self.rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.item_factors = self.rng.normal(0, 0.1, size=(n_items, self.n_factors))

        I_f = np.eye(self.n_factors) * self.reg_lambda

        for _ in range(self.n_epochs):
            # ---- Fix item factors, solve for user factors ----
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                row = C.getrow(u)
                idx = row.indices
                if len(idx) == 0:
                    continue
                Cu = row.data
                Yu = self.item_factors[idx]
                A = YtY + (Yu.T * (Cu - 1.0)) @ Yu + I_f
                b = (Yu.T * Cu) @ np.ones(len(idx))  # since p_u = 1 for observed entries
                self.user_factors[u] = np.linalg.solve(A, b)

            # ---- Fix user factors, solve for item factors ----
            C_csc = C.tocsc()
            XtX = self.user_factors.T @ self.user_factors
            for i in range(n_items):
                col = C_csc.getcol(i)
                idx = col.indices
                if len(idx) == 0:
                    continue
                Ci = col.data
                Xi = self.user_factors[idx]
                A = XtX + (Xi.T * (Ci - 1.0)) @ Xi + I_f
                b = (Xi.T * Ci) @ np.ones(len(idx))
                self.item_factors[i] = np.linalg.solve(A, b)

        return self

    def score_all_items(self, user_id) -> np.ndarray:
        """Vectorized dot-product of a user's latent vector against every item vector."""
        if user_id not in self.user_index_:
            return np.zeros(len(self.item_index_))
        u_idx = self.user_index_[user_id]
        return self.user_factors[u_idx] @ self.item_factors.T

    def item_ids_ordered(self) -> np.ndarray:
        return np.array(list(self.item_index_.keys()))


def build_interaction_matrix(submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw submissions into a (user_id, question_id, weight) implicit
    feedback table: an attempt = 1.0, a solved attempt bumps the weight to 3.0
    (solving signals much stronger positive affinity than merely attempting).
    """
    sub = submissions_df.copy()
    sub["is_accepted"] = (sub["status"] == "Accepted").astype(int)
    agg = (
        sub.groupby(["user_id", "question_id"])["is_accepted"]
        .agg(["count", "max"])
        .rename(columns={"count": "attempts", "max": "ever_solved"})
        .reset_index()
    )
    agg["weight"] = 1.0 + 2.0 * agg["ever_solved"]  # 1.0 attempted-only, 3.0 solved
    return agg[["user_id", "question_id", "weight"]]


# ==============================================================================
# 2. Content-Based Similarity (cold-start fallback)
# ==============================================================================

class ContentBasedRecommender:
    """Tag-aware similarity using the question topic ontology rather than raw text noise."""

    def __init__(self, questions_df: pd.DataFrame):
        self.questions_df = questions_df.reset_index(drop=True)
        all_tags = sorted({tag for tags in self.questions_df["topic_tags"] for tag in tags})
        self.tag_index = {tag: i for i, tag in enumerate(all_tags)}
        self.tag_matrix = np.zeros((len(self.questions_df), len(all_tags)), dtype=float)

        for row_idx, tags in enumerate(self.questions_df["topic_tags"]):
            for tag in tags:
                if tag in self.tag_index:
                    self.tag_matrix[row_idx, self.tag_index[tag]] = 1.0

        self.qid_to_row = {qid: i for i, qid in enumerate(self.questions_df["question_id"])}

    def _question_tag_vector(self, question_id: int) -> np.ndarray:
        row = self.qid_to_row.get(question_id)
        if row is None:
            return np.zeros(self.tag_matrix.shape[1])
        return self.tag_matrix[row]

    def similar_to_user_profile(self, solved_question_ids: List[int], failed_question_ids: List[int] | None = None) -> np.ndarray:
        """Score every question by how closely its tags match the user's solved + weak-area profile."""
        profile = np.zeros(self.tag_matrix.shape[1], dtype=float)

        for qid in solved_question_ids:
            profile += self._question_tag_vector(qid)

        if failed_question_ids:
            for qid in failed_question_ids:
                profile += 0.75 * self._question_tag_vector(qid)

        if not np.any(profile):
            return np.zeros(len(self.questions_df))

        q_norms = np.linalg.norm(self.tag_matrix, axis=1)
        profile_norm = np.linalg.norm(profile)
        if profile_norm < 1e-12:
            return np.zeros(len(self.questions_df))

        scores = (self.tag_matrix @ profile) / (q_norms * profile_norm + 1e-12)
        return np.clip(scores, 0.0, 1.0)


# ==============================================================================
# 3. Hybrid Recommender (public interface)
# ==============================================================================

class HybridRecommender:
    """
    Combines ALS collaborative-filtering scores with content-based scores:

        final_score = w_cf * cf_score + w_content * content_score

    Both score vectors are min-max normalized before blending so neither
    signal dominates purely due to differing scales.
    """

    def __init__(self, questions_df: pd.DataFrame, submissions_df: pd.DataFrame):
        self.questions_df = questions_df.reset_index(drop=True)
        interactions = build_interaction_matrix(submissions_df)
        self.als = ALSMatrixFactorization().fit(interactions)
        self.content_model = ContentBasedRecommender(questions_df)
        self.submissions_df = submissions_df

    @staticmethod
    def _min_max(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    def recommend(
        self,
        user_id: int,
        top_n: int = config.TOP_N_RECOMMENDATIONS,
        topic_profile: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        item_ids = self.als.item_ids_ordered()
        cf_scores_raw = self.als.score_all_items(user_id)

        # Map ALS's internal item ordering back onto the full questions_df ordering.
        cf_score_by_qid = dict(zip(item_ids, cf_scores_raw))
        cf_scores = self.questions_df["question_id"].map(cf_score_by_qid).fillna(0.0).to_numpy()

        solved_ids = self.submissions_df.loc[
            (self.submissions_df["user_id"] == user_id) & (self.submissions_df["status"] == "Accepted"),
            "question_id",
        ].unique().tolist()
        failed_ids = self.submissions_df.loc[
            (self.submissions_df["user_id"] == user_id) & (self.submissions_df["status"] != "Accepted"),
            "question_id",
        ].unique().tolist()

        content_scores = self.content_model.similar_to_user_profile(solved_ids, failed_ids)
        blended = (
            config.HYBRID_CF_WEIGHT * self._min_max(cf_scores)
            + config.HYBRID_CONTENT_WEIGHT * self._min_max(content_scores)
        )

        # Incorporate Topic Weakness Profile Boost
        if topic_profile is not None and not topic_profile.empty:
            weak_map = topic_profile.set_index("topic")["risk_score"].to_dict()
            boosts = np.zeros(len(self.questions_df))
            for idx, tags in enumerate(self.questions_df["topic_tags"]):
                tag_boost = max([weak_map.get(tag, 0.0) for tag in tags] + [0.0])
                boosts[idx] = tag_boost
            blended += 0.85 * boosts

        if not np.any(blended > 0):
            fallback = self.questions_df["acceptance_rate"].fillna(0.5).to_numpy()
            if fallback.max() > 0:
                fallback = fallback / fallback.max()
                blended = 0.5 * fallback + 0.5 * self._min_max(np.array([1.0 if d == "Easy" else 0.65 if d == "Medium" else 0.35 for d in self.questions_df["difficulty"]]))

        result = self.questions_df.copy()
        result["recommendation_score"] = blended

        # NEVER recommend a question the user has already solved.
        candidates = result[~result["question_id"].isin(solved_ids)].copy()
        if len(candidates) < top_n:
            candidates = result.copy()

        candidates = candidates.sort_values("recommendation_score", ascending=False, kind="mergesort")

        # Enforce diversity across topic tags (avoid 5 identical questions)
        selected_rows = []
        selected_qids = set()
        topic_counts = {}

        for _, row in candidates.iterrows():
            qid = row["question_id"]
            if qid in selected_qids:
                continue

            tags = row.get("topic_tags", [])
            primary_tag = tags[0] if tags else "General"
            
            # Allow at most 2 recommendations per primary topic tag for diversity
            if topic_counts.get(primary_tag, 0) >= 2 and len(selected_rows) < top_n - 1:
                continue

            selected_rows.append(row)
            selected_qids.add(qid)
            topic_counts[primary_tag] = topic_counts.get(primary_tag, 0) + 1

            if len(selected_rows) >= top_n:
                break

        if len(selected_rows) < top_n:
            remaining = candidates[~candidates["question_id"].isin(selected_qids)].head(top_n - len(selected_rows))
            selected_rows.extend([r for _, r in remaining.iterrows()])

        top_df = pd.DataFrame(selected_rows).head(top_n).copy()

        def _generate_reason(row):
            tags = row.get("topic_tags", [])
            diff = row.get("difficulty", "Medium")

            if topic_profile is not None and not topic_profile.empty and tags:
                for tag in tags:
                    match = topic_profile[topic_profile["topic"] == tag]
                    if not match.empty:
                        t_info = match.iloc[0]
                        weak_lvl = t_info.get("weakness_level", "Neutral")
                        attempts = t_info.get("attempts", 0)
                        success_pct = t_info.get("success_rate_pct", "0%")

                        if weak_lvl in ["Critical", "Weak"]:
                            return (
                                f"{tag} is currently your highest-risk topic ({attempts} attempts, {success_pct} success rate), "
                                f"so this {diff} problem is appropriate for targeted practice."
                            )
                        elif weak_lvl == "Moderate":
                            return (
                                f"{tag} is a moderate weakness ({attempts} attempts, {success_pct} success rate). "
                                f"Recommended to build accuracy and speed."
                            )
                        elif t_info.get("coverage_level") == "Low":
                            return (
                                f"Low coverage in {tag} ({t_info.get('unique_solved', 0)} solved of {t_info.get('total_available', 0)} available). "
                                f"Recommended to fill topic coverage gap."
                            )

            if tags:
                return f"Recommended {diff} problem in {tags[0]} for skill building."
            return f"Recommended {diff} problem for practice."

        top_df["reason"] = top_df.apply(_generate_reason, axis=1)
        return top_df[["question_id", "title", "difficulty", "topic_tags", "recommendation_score", "reason"]]


if __name__ == "__main__":
    from data_processing import generate_full_synthetic_dataset

    users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
    recommender = HybridRecommender(questions_df, submissions_df)

    sample_user_id = int(users_df["user_id"].iloc[0])
    top5 = recommender.recommend(sample_user_id, top_n=config.TOP_N_RECOMMENDATIONS)
    print(f"Top {len(top5)} recommendations for user {sample_user_id}:")
    print(top5.to_string(index=False))
