"""
recommender.py
--------------
RECOMMENDATION ENGINE

Architecture:
  1. Collaborative Filtering via implicit-feedback Matrix Factorization (ALS):
     - OFFLINE: Trained on multi-user synthetic interactions across canonical questions.
       Learned item factors Y and precomputed Y^T Y are saved to models/als_model.npz.
     - ONLINE: Real user interactions are folded in on the fly via closed-form ridge solve:
         x_u = (Y^T Cu Y + lambda*I)^-1 Y^T Cu p_u
       without retraining the model or modifying item factors.
     - Cold-start guard: If user interactions < 3, ALS is marked unavailable and recommendation
       relies on content-based similarity and weakness profiling.
  2. Content-Based Filtering:
     - Precomputed canonical topic tag similarity against user's solved/failed profile.
     - Guaranteed non-empty recommendations even with 0 collaborative interactions.
  3. Profile-Guided Hybrid Blending:
     - Blends normalized ALS scores + content similarity + topic weakness risk scores.
     - Strictly excludes solved problems, enforces topic diversity, and generates transparent reasons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

import config

# ==============================================================================
# 1. Implicit-Feedback ALS Matrix Factorization with New-User Fold-In
# ==============================================================================

class ALSMatrixFactorization:
    """
    Alternating Least Squares for implicit feedback (Hu, Koren & Volinsky, 2008).
    Supports offline training, serialization, and online closed-form user fold-in.
    """

    def __init__(
        self,
        n_factors: int = config.MF_LATENT_DIM,
        reg_lambda: float = config.MF_REG_LAMBDA,
        alpha: float = config.MF_CONFIDENCE_ALPHA,
        n_epochs: int = config.MF_EPOCHS,
        random_state: int = config.RANDOM_SEED,
        latent_dim: int | None = None,
        epochs: int | None = None,
    ):
        if latent_dim is not None:
            n_factors = latent_dim
        if epochs is not None:
            n_epochs = epochs
        self.n_factors = n_factors
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(random_state)

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_index_: dict | None = None
        self.item_index_: dict | None = None
        self.YtY_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.item_factors is not None

    @property
    def latent_dim(self) -> int:
        return self.n_factors

    @property
    def YtY(self) -> np.ndarray | None:
        if self.YtY_ is not None:
            return self.YtY_
        if self.item_factors is not None:
            return self.item_factors.T @ self.item_factors
        return None

    @property
    def item_ids(self) -> list:
        return list(self.item_index_.keys()) if self.item_index_ is not None else []

    def fit(self, interactions: pd.DataFrame, user_col="user_id", item_col="question_id", weight_col="weight"):
        """
        Offline training on multi-user interaction dataset.
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
        C = R.copy()
        C.data = 1.0 + self.alpha * C.data

        self.user_factors = self.rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.item_factors = self.rng.normal(0, 0.1, size=(n_items, self.n_factors))

        I_f = np.eye(self.n_factors) * self.reg_lambda

        for _ in range(self.n_epochs):
            # Fix item factors, solve user factors
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                row = C.getrow(u)
                idx = row.indices
                if len(idx) == 0:
                    continue
                Cu = row.data
                Yu = self.item_factors[idx]
                A = YtY + (Yu.T * (Cu - 1.0)) @ Yu + I_f
                b = (Yu.T * Cu) @ np.ones(len(idx))
                self.user_factors[u] = np.linalg.solve(A, b)

            # Fix user factors, solve item factors
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

        self.YtY_ = self.item_factors.T @ self.item_factors
        return self

    # ---- Serialization ----------------------------------------------------------
    def save(self, file_path: str | Path = config.ALS_MODEL_PATH) -> Path:
        """Saves item factors and mappings to disk for fast online fold-in."""
        if self.item_factors is None or self.item_index_ is None:
            raise RuntimeError("Cannot save untrained ALS model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        item_ids = np.array(list(self.item_index_.keys()))
        np.savez_compressed(
            path,
            item_factors=self.item_factors,
            item_ids=item_ids,
            yt_y=self.YtY_ if self.YtY_ is not None else (self.item_factors.T @ self.item_factors),
            alpha=self.alpha,
            reg_lambda=self.reg_lambda,
            n_factors=self.n_factors,
        )
        return path

    def load(self, file_path: str | Path = config.ALS_MODEL_PATH) -> ALSMatrixFactorization:
        """Loads learned item factors from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ALS model artifact not found at '{path}'. "
                "Please run offline training first: python training/train_models.py"
            )
        data = np.load(path, allow_pickle=True)
        self.item_factors = data["item_factors"]
        item_ids = data["item_ids"]
        self.item_index_ = {it: i for i, it in enumerate(item_ids)}
        self.YtY_ = data["yt_y"] if "yt_y" in data else (self.item_factors.T @ self.item_factors)
        self.alpha = float(data["alpha"]) if "alpha" in data else self.alpha
        self.reg_lambda = float(data["reg_lambda"]) if "reg_lambda" in data else self.reg_lambda
        self.n_factors = int(data["n_factors"]) if "n_factors" in data else self.n_factors
        return self

    # ---- Online Fold-In for New Real User ---------------------------------------
    def fold_in_user(
        self,
        observed_items: List[int] | pd.DataFrame,
        observed_weights: List[float] | None = None,
    ) -> Tuple[np.ndarray, bool] | np.ndarray:
        """
        Folds in a new real user using fixed offline item factors Y:
            x_u = (Y^T Y + Y_u^T (C_u - 1) Y_u + lambda*I)^-1 Y_u^T C_u p_u
        Supports passing either:
          - (observed_items_list, observed_weights_list) -> returns (scores, is_available)
          - interactions DataFrame with ['question_id', 'weight'] -> returns scores
        """
        if isinstance(observed_items, pd.DataFrame):
            df = observed_items
            qids = df["question_id"].tolist() if "question_id" in df.columns else []
            weights = df["weight"].tolist() if "weight" in df.columns else [1.0] * len(qids)
            scores, _ = self._fold_in_user_core(qids, weights)
            return scores

        return self._fold_in_user_core(observed_items, observed_weights)

    def _fold_in_user_core(
        self,
        observed_items: List[int],
        observed_weights: List[float] | None = None,
    ) -> Tuple[np.ndarray, bool]:
        if self.item_factors is None or self.item_index_ is None:
            return np.zeros(0), False

        unique_interactions: dict[int, float] = {}
        for i, qid in enumerate(observed_items):
            if qid in self.item_index_:
                idx = self.item_index_[qid]
                w = observed_weights[i] if observed_weights and i < len(observed_weights) else 1.0
                unique_interactions[idx] = unique_interactions.get(idx, 0.0) + w

        # Cold-start guard: require at least 3 unique valid interactions for meaningful collaborative latent vector
        if len(unique_interactions) < 3:
            return np.zeros(len(self.item_index_)), False

        idx = np.array(list(unique_interactions.keys()))
        weights = np.array(list(unique_interactions.values()))
        Cu = 1.0 + self.alpha * weights
        Yu = self.item_factors[idx]

        I_f = np.eye(self.n_factors) * self.reg_lambda
        YtY = self.YtY_ if self.YtY_ is not None else (self.item_factors.T @ self.item_factors)

        # Closed-form ridge solve
        A = YtY + (Yu.T * (Cu - 1.0)) @ Yu + I_f
        b = (Yu.T * Cu) @ np.ones(len(idx))
        x_user = np.linalg.solve(A, b)

        scores = x_user @ self.item_factors.T
        return scores, True

    def score_all_items(self, user_id: int, user_interactions: pd.DataFrame | None = None) -> np.ndarray:
        """Scores all items for user_id via fold-in or pre-computed user factors."""
        if self.user_index_ is not None and user_id in self.user_index_ and self.user_factors is not None:
            u_idx = self.user_index_[user_id]
            return self.user_factors[u_idx] @ self.item_factors.T

        if user_interactions is not None and not user_interactions.empty:
            qids = user_interactions["question_id"].tolist()
            weights = user_interactions["weight"].tolist() if "weight" in user_interactions.columns else None
            scores, ok = self.fold_in_user(qids, weights)
            if ok:
                return scores

        return np.zeros(len(self.item_index_)) if self.item_index_ else np.zeros(0)

    def item_ids_ordered(self) -> np.ndarray:
        return np.array(list(self.item_index_.keys())) if self.item_index_ else np.array([])


def build_interaction_matrix(submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw submissions into a (user_id, question_id, weight) implicit feedback table:
    Attempted = 1.0, Solved = 3.0.
    """
    if submissions_df.empty:
        return pd.DataFrame(columns=["user_id", "question_id", "weight"])
    sub = submissions_df.copy()
    sub["is_accepted"] = (sub["status"] == "Accepted").astype(int)
    agg = (
        sub.groupby(["user_id", "question_id"])["is_accepted"]
        .agg(["count", "max"])
        .rename(columns={"count": "attempts", "max": "ever_solved"})
        .reset_index()
    )
    agg["weight"] = 1.0 + 2.0 * agg["ever_solved"]
    return agg[["user_id", "question_id", "weight"]]


# ==============================================================================
# 2. Content-Based Similarity (cold-start & profile matching)
# ==============================================================================

class ContentBasedRecommender:
    """Topic-tag based cosine similarity matching candidate questions to user history."""

    def __init__(self, questions_df: pd.DataFrame):
        self.questions_df = questions_df.reset_index(drop=True)
        all_tags = sorted({tag for tags in self.questions_df["topic_tags"] for tag in tags if tag})
        self.tag_index = {tag: i for i, tag in enumerate(all_tags)}
        n_tags = max(len(all_tags), 1)
        self.tag_matrix = np.zeros((len(self.questions_df), n_tags), dtype=float)

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

    def similar_to_user_profile(
        self,
        solved_question_ids: List[int],
        failed_question_ids: List[int] | None = None,
    ) -> np.ndarray:
        """Scores candidate questions against user's solved and weak topic profile."""
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
# 3. Hybrid Recommender (online inference & evaluation)
# ==============================================================================

class HybridRecommender:
    """
    Blends collaborative filtering (ALS fold-in), content-based tag similarity,
    and structured topic weakness boosting.
    """

    def __init__(
        self,
        questions_df: pd.DataFrame,
        submissions_df: pd.DataFrame | None = None,
        als_model: ALSMatrixFactorization | None = None,
    ):
        self.questions_df = questions_df.reset_index(drop=True)
        self.submissions_df = submissions_df if submissions_df is not None else pd.DataFrame(columns=config.SUBMISSION_COLUMNS)
        self.content_model = ContentBasedRecommender(self.questions_df)

        if als_model is not None:
            self.als = als_model
        elif config.ALS_MODEL_PATH.exists():
            self.als = ALSMatrixFactorization().load(config.ALS_MODEL_PATH)
        elif self.submissions_df is not None and len(self.submissions_df) > 100:
            interactions = build_interaction_matrix(self.submissions_df)
            self.als = ALSMatrixFactorization().fit(interactions)
        else:
            self.als = ALSMatrixFactorization()

    @staticmethod
    def _min_max(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    def recommend(
        self,
        user_id: int,
        top_n: int = config.TOP_N_RECOMMENDATIONS,
        topic_profile: pd.DataFrame | None = None,
        user_submissions_df: pd.DataFrame | None = None,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        if top_k is not None:
            top_n = top_k
        subs = user_submissions_df if user_submissions_df is not None else self.submissions_df
        user_subs = subs[subs["user_id"] == user_id]

        solved_ids = user_subs.loc[user_subs["status"] == "Accepted", "question_id"].unique().tolist()
        failed_ids = user_subs.loc[user_subs["status"] != "Accepted", "question_id"].unique().tolist()

        # 1. ALS Collaborative Filtering via Fold-In
        cf_available = False
        cf_scores = np.zeros(len(self.questions_df))

        if self.als is not None and self.als.item_factors is not None and not user_subs.empty:
            interactions = build_interaction_matrix(user_subs)
            qids = interactions["question_id"].tolist()
            weights = interactions["weight"].tolist()
            raw_cf, cf_available = self.als.fold_in_user(qids, weights)
            if cf_available:
                item_ids = self.als.item_ids_ordered()
                cf_score_by_qid = dict(zip(item_ids, raw_cf))
                cf_scores = self.questions_df["question_id"].map(cf_score_by_qid).fillna(0.0).to_numpy()

        # 2. Content-based similarity
        content_scores = self.content_model.similar_to_user_profile(solved_ids, failed_ids)

        # 3. Topic Weakness Boost
        if topic_profile is None and not user_subs.empty:
            from data_processing import compute_user_topic_profile
            topic_profile = compute_user_topic_profile(user_subs, self.questions_df, user_id=user_id)

        weakness_boost = np.zeros(len(self.questions_df))
        if topic_profile is not None and not topic_profile.empty:
            weak_map = topic_profile.set_index("topic")["risk_score"].to_dict()
            for idx, tags in enumerate(self.questions_df["topic_tags"]):
                tag_boost = max([weak_map.get(tag, 0.0) for tag in tags] + [0.0])
                weakness_boost[idx] = tag_boost

        # 4. Normalized Blending
        cf_norm = self._min_max(cf_scores)
        content_norm = self._min_max(content_scores)
        weakness_norm = self._min_max(weakness_boost)

        if cf_available:
            blended = (
                config.HYBRID_CF_WEIGHT * cf_norm
                + config.HYBRID_CONTENT_WEIGHT * content_norm
                + config.HYBRID_WEAKNESS_WEIGHT * weakness_norm
            )
        else:
            # Cold start: dynamically re-weight content and weakness
            blended = (
                (config.HYBRID_CF_WEIGHT + config.HYBRID_CONTENT_WEIGHT) * content_norm
                + config.HYBRID_WEAKNESS_WEIGHT * weakness_norm
            )

        # 5. Global difficulty suitability fallback if user has no signal
        if not np.any(blended > 0):
            fallback_acc = self.questions_df["acceptance_rate"].fillna(0.5).to_numpy()
            diff_score = np.array([
                1.0 if d == "Easy" else 0.65 if d == "Medium" else 0.35
                for d in self.questions_df["difficulty"]
            ])
            blended = 0.5 * self._min_max(fallback_acc) + 0.5 * self._min_max(diff_score)

        result = self.questions_df.copy()
        result["recommendation_score"] = blended

        # NEVER recommend problems the user has already solved
        candidates = result[~result["question_id"].isin(solved_ids)].copy()
        if len(candidates) < top_n:
            candidates = result.copy()

        candidates = candidates.sort_values("recommendation_score", ascending=False, kind="mergesort")

        # Enforce topic diversity (max 2 recommendations per primary tag)
        selected_rows = []
        selected_qids = set()
        topic_counts: Dict[str, int] = {}

        for _, row in candidates.iterrows():
            qid = row["question_id"]
            if qid in selected_qids:
                continue

            tags = row.get("topic_tags", [])
            primary_tag = tags[0] if tags else "General"

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

        # Generate transparent, evidence-grounded reasons
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
                                f"{tag} is currently flagged as {weak_lvl} ({attempts} attempts, {success_pct} success rate). "
                                f"Recommended {diff} problem for targeted weakness reinforcement."
                            )
                        elif weak_lvl == "Moderate":
                            return (
                                f"{tag} is a moderate growth area ({attempts} attempts, {success_pct} success rate). "
                                f"Recommended to build speed and confidence."
                            )

            if cf_available:
                return f"Collaborative filtering candidate ({diff} problem in {', '.join(tags[:2])}) aligning with peer learning paths."
            if tags:
                return f"Recommended {diff} problem in {tags[0]} based on your problem-solving profile."
            return f"Recommended {diff} problem for algorithmic practice."

        top_df["reason"] = top_df.apply(_generate_reason, axis=1)
        return top_df[["question_id", "title", "difficulty", "topic_tags", "recommendation_score", "reason"]]

    # ---- Recommendation Evaluation ----------------------------------------------
    @staticmethod
    def evaluate_recommendations(
        recommender: HybridRecommender,
        user_ids: List[int],
        history_subs: pd.DataFrame,
        future_subs: pd.DataFrame,
        top_n: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluates recommendations on held-out temporal future interactions.
        Input: history_subs (used by recommender)
        Ground Truth: future_subs (problems solved by user in the future)
        Reports Precision@K, Recall@K, Hit Rate@K, NDCG@K.
        """
        precisions = []
        recalls = []
        hits = []
        ndcgs = []

        future_solved_by_user = (
            future_subs[future_subs["status"] == "Accepted"]
            .groupby("user_id")["question_id"]
            .apply(set)
            .to_dict()
        )
        history_solved_by_user = (
            history_subs[history_subs["status"] == "Accepted"]
            .groupby("user_id")["question_id"]
            .apply(set)
            .to_dict()
        )

        for uid in user_ids:
            # Positive ground truth: questions newly solved in the future (excluding already-solved in history)
            past_solved = history_solved_by_user.get(uid, set())
            gt_solved = future_solved_by_user.get(uid, set()) - past_solved
            if not gt_solved:
                continue

            u_hist = history_subs[history_subs["user_id"] == uid]
            recs_df = recommender.recommend(user_id=uid, top_n=top_n, user_submissions_df=u_hist)
            rec_qids = recs_df["question_id"].tolist()

            matched = [1 if q in gt_solved else 0 for q in rec_qids]
            n_matched = sum(matched)

            prec = n_matched / float(top_n)
            rec = n_matched / float(len(gt_solved))
            hit = 1.0 if n_matched > 0 else 0.0

            dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(matched)])
            # Ideal DCG corresponds to placing up to min(top_n, |gt_solved|) relevant items at top ranks
            ideal_hits = min(top_n, len(gt_solved))
            idcg = sum([1.0 / np.log2(idx + 2) for idx in range(ideal_hits)])
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            hits.append(hit)
            ndcgs.append(ndcg)

        return {
            f"precision@{top_n}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall@{top_n}": float(np.mean(recalls)) if recalls else 0.0,
            f"hit_rate@{top_n}": float(np.mean(hits)) if hits else 0.0,
            f"ndcg@{top_n}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "evaluated_users": len(precisions),
        }


evaluate_recommendations = HybridRecommender.evaluate_recommendations


if __name__ == "__main__":
    from data_processing import generate_full_synthetic_dataset

    users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
    recommender = HybridRecommender(questions_df, submissions_df)

    sample_user_id = int(users_df["user_id"].iloc[0])
    top5 = recommender.recommend(sample_user_id, top_n=config.TOP_N_RECOMMENDATIONS)
    print(f"Top {len(top5)} recommendations for user {sample_user_id}:")
    print(top5.to_string(index=False))

