"""
data_processing.py
-------------------
DATA ENGINE

Responsibilities:
  1. Generate a realistic, internally-consistent synthetic dataset
     (questions, users, submissions) that mirrors a real LeetCode data
     export -- this is the drop-in replacement point for a real scrape/API dump.
  2. Run a fully-vectorized feature engineering pipeline that turns raw
     submission logs into a per-user feature matrix consumed by:
       - nlp_cluster.py   (needs raw failed-submission descriptions)
       - recommender.py   (needs the user-item interaction matrix)
       - predictor.py      (needs the aggregated numeric feature matrix)

No per-row Python loops are used for numeric aggregation -- everything is
expressed as pandas groupby/pivot/merge or numpy broadcasting so it scales
to real submission volumes (10^5 - 10^7 rows) without a rewrite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from schemas import QUESTION_COLUMNS, USER_COLUMNS, SUBMISSION_COLUMNS


# ==============================================================================
# 1. SYNTHETIC DATA GENERATION
# ==============================================================================

_DESCRIPTION_TEMPLATES = [
    "Given {a}, determine {b} using an efficient {c} approach while respecting {d} constraints.",
    "You are given {a}. Return {b}. Your solution should leverage {c} and consider {d}.",
    "Design an algorithm that, given {a}, computes {b} in optimal time via {c}, mindful of {d}.",
    "Implement a function that receives {a} and outputs {b}, optimized with {c} under {d} limits.",
]

_SUBJECTS = ["an array of integers", "a binary tree", "a weighted graph", "a string s",
             "a matrix of size m x n", "a linked list", "a set of intervals", "two sorted arrays"]
_GOALS = ["the maximum subarray sum", "whether a valid path exists", "the shortest path length",
          "the number of connected components", "the longest palindromic substring",
          "the k-th largest element", "a valid topological ordering", "the minimum edit distance"]


def _rng() -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_SEED)


def generate_synthetic_questions(n: int = config.NUM_QUESTIONS) -> pd.DataFrame:
    """
    Builds a synthetic `questions` table.

    Each question gets 1-3 topic tags and a template-generated description
    that textually references its tags -- this keeps the NLP embeddings in
    nlp_cluster.py semantically meaningful (tag words actually appear in
    the text sentence-transformers will embed).
    """
    rng = _rng()
    difficulties = rng.choice(config.DIFFICULTIES, size=n, p=config.DIFFICULTY_WEIGHTS)

    tags_per_question = rng.integers(1, 4, size=n)
    topic_tags_col = [
        list(rng.choice(config.TOPIC_TAGS, size=k, replace=False))
        for k in tags_per_question
    ]

    templates = rng.choice(_DESCRIPTION_TEMPLATES, size=n)
    subjects = rng.choice(_SUBJECTS, size=n)
    goals = rng.choice(_GOALS, size=n)

    descriptions = [
        templates[i].format(a=subjects[i], b=goals[i], c=", ".join(topic_tags_col[i]),
                             d=difficulties[i].lower() + "-tier time/space")
        for i in range(n)
    ]

    # Acceptance rate is inversely related to difficulty (vectorized, not per-row branching).
    difficulty_penalty = pd.Series(difficulties).map(config.DIFFICULTY_SCORE).to_numpy()
    acceptance_rate = np.clip(0.75 - 0.12 * difficulty_penalty + rng.normal(0, 0.05, n), 0.05, 0.95)

    df = pd.DataFrame({
        "question_id": np.arange(1, n + 1),
        "title": [f"Problem {i}" for i in range(1, n + 1)],
        "description": descriptions,
        "difficulty": difficulties,
        "topic_tags": topic_tags_col,
        "acceptance_rate": acceptance_rate,
    })
    return df[QUESTION_COLUMNS]


def generate_synthetic_users(n: int = config.NUM_USERS) -> pd.DataFrame:
    """
    Builds a synthetic `users` table with a hidden `latent_skill` variable.

    latent_skill is the generative "ground truth" ability score that drives
    (a) submission correctness probability and (b) the contest_rating target.
    It is NEVER passed to the ML models -- it exists purely so the synthetic
    dataset has a coherent underlying signal for the models to rediscover
    from observable features, exactly like a real learner's true skill drives
    their observable submission history.
    """
    rng = _rng()
    latent_skill = rng.normal(loc=0.0, scale=1.0, size=n)
    account_age_days = rng.integers(30, 1500, size=n)

    # Contest rating: baseline + skill effect + mild experience effect + noise.
    contest_rating = (
        1200
        + 260 * latent_skill
        + 0.08 * np.sqrt(account_age_days) * 10
        + rng.normal(0, 60, n)
    )
    contest_rating = np.clip(contest_rating, 800, 3000)

    df = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "account_age_days": account_age_days,
        "latent_skill": latent_skill,
        "contest_rating": contest_rating,
    })
    return df[USER_COLUMNS]


def generate_synthetic_submissions(
    users_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    n_submissions: int = config.NUM_SUBMISSIONS,
) -> pd.DataFrame:
    """
    Builds a synthetic `submissions` log.

    Correctness probability is modeled with a vectorized logistic function of
    (user latent_skill - question difficulty_score), i.e. classic item-response
    theory. This produces realistic patterns: strong users still occasionally
    fail Hard problems, weak users occasionally fluke an Easy one.
    """
    rng = _rng()

    user_ids = rng.integers(1, len(users_df) + 1, size=n_submissions)
    question_ids = rng.integers(1, len(questions_df) + 1, size=n_submissions)

    skill_lookup = users_df.set_index("user_id")["latent_skill"].to_numpy()
    difficulty_lookup = questions_df.set_index("question_id")["difficulty"].map(config.DIFFICULTY_SCORE).to_numpy()

    user_skill = skill_lookup[user_ids - 1]
    question_difficulty = difficulty_lookup[question_ids - 1]

    # Item-response-theory style solve probability (fully vectorized sigmoid).
    logits = 1.4 * (user_skill - (question_difficulty - 2.2))
    solve_prob = 1.0 / (1.0 + np.exp(-logits))

    is_accepted = rng.binomial(1, np.clip(solve_prob, 0.02, 0.98))

    # For failed attempts, distribute across the non-Accepted statuses.
    fail_statuses = config.SUBMISSION_STATUSES[1:]
    fail_weights = config.STATUS_BASE_WEIGHTS[1:] / config.STATUS_BASE_WEIGHTS[1:].sum()
    fail_choice = rng.choice(fail_statuses, size=n_submissions, p=fail_weights)

    status = np.where(is_accepted == 1, "Accepted", fail_choice)

    # Timestamps: skew recent activity slightly heavier (power-law over the window).
    days_ago = (rng.power(1.5, size=n_submissions) * config.DATASET_WINDOW_DAYS).astype(int)
    timestamps = pd.Timestamp.now().normalize() - pd.to_timedelta(days_ago, unit="D")

    runtime_ms = rng.gamma(shape=2.0, scale=45.0, size=n_submissions)
    language = rng.choice(config.LANGUAGES, size=n_submissions, p=[0.5, 0.2, 0.15, 0.1, 0.05])

    df = pd.DataFrame({
        "user_id": user_ids,
        "question_id": question_ids,
        "timestamp": timestamps,
        "status": status,
        "runtime_ms": runtime_ms,
        "language": language,
    }).sort_values("timestamp").reset_index(drop=True)

    return df[SUBMISSION_COLUMNS]


def generate_full_synthetic_dataset():
    """Convenience entry point returning (users_df, questions_df, submissions_df)."""
    questions_df = generate_synthetic_questions()
    users_df = generate_synthetic_users()
    submissions_df = generate_synthetic_submissions(users_df, questions_df)
    return users_df, questions_df, submissions_df


# ==============================================================================
# 2. FEATURE ENGINEERING PIPELINE
# ==============================================================================

class FeatureEngineer:
    """
    Transforms raw (users, questions, submissions) tables into a single
    per-user numeric feature matrix.

    Every computation below is a groupby / pivot / merge -- there is no
    row-wise Python loop, so this scales linearly with vectorized pandas
    execution rather than interpreter overhead.
    """

    def __init__(self, users_df: pd.DataFrame, questions_df: pd.DataFrame, submissions_df: pd.DataFrame):
        self.users_df = users_df
        self.questions_df = questions_df
        self.submissions_df = submissions_df.merge(
            questions_df[["question_id", "difficulty", "topic_tags", "acceptance_rate"]],
            on="question_id", how="left",
        )

    # ---- 2a. Accuracy ratios -------------------------------------------------
    def compute_accuracy_ratios(self) -> pd.DataFrame:
        """Overall and per-difficulty accuracy ratio, one row per user."""
        sub = self.submissions_df
        sub = sub.assign(is_accepted=(sub["status"] == "Accepted").astype(int))

        overall = (
            sub.groupby("user_id")["is_accepted"]
            .agg(total_submissions="count", total_accepted="sum")
            .assign(overall_accuracy=lambda d: d["total_accepted"] / d["total_submissions"])
        )

        per_difficulty = (
            sub.pivot_table(
                index="user_id", columns="difficulty", values="is_accepted",
                aggfunc="mean", fill_value=0.0,
            )
            .add_prefix("accuracy_")
            .rename(columns=lambda c: c.lower())
        )

        return overall.join(per_difficulty, how="left").fillna(0.0)

    # ---- 2b. Difficulty distribution -----------------------------------------
    def compute_difficulty_distribution(self) -> pd.DataFrame:
        """Share of a user's total attempts spent on Easy/Medium/Hard problems."""
        sub = self.submissions_df
        counts = pd.pivot_table(
            sub, index="user_id", columns="difficulty", values="question_id",
            aggfunc="count", fill_value=0,
        )
        shares = counts.div(counts.sum(axis=1), axis=0).add_prefix("share_").rename(columns=lambda c: c.lower())
        return shares

    # ---- 2c. Recency / momentum ----------------------------------------------
    def compute_recency_momentum(self, half_life_days: float = config.RECENCY_HALF_LIFE_DAYS) -> pd.DataFrame:
        """
        Exponentially time-decayed "momentum" score per user:

            momentum = sum(weight_i * is_accepted_i) / sum(weight_i)
            weight_i = 0.5 ** (days_ago_i / half_life_days)

        This rewards users who are *currently* solving well over users whose
        accuracy was good months ago but has since stalled -- much stronger
        signal for predicting a near-term contest rating than a flat average.
        """
        sub = self.submissions_df.copy()
        now = sub["timestamp"].max()
        days_ago = (now - sub["timestamp"]).dt.total_seconds() / 86400.0
        weight = np.power(0.5, days_ago / half_life_days)

        sub["_weight"] = weight
        sub["_weighted_correct"] = weight * (sub["status"] == "Accepted").astype(int)

        momentum = (
            sub.groupby("user_id")
            .apply(lambda d: d["_weighted_correct"].sum() / d["_weight"].sum(), include_groups=False)
            .rename("recency_momentum")
            .to_frame()
        )

        # Submission frequency in the last 2x half-life window as an activity-level signal.
        recent_mask = days_ago <= (2 * half_life_days)
        recent_activity = (
            sub[recent_mask].groupby("user_id").size().rename("recent_submission_count").to_frame()
        )

        return momentum.join(recent_activity, how="left").fillna(0.0)

    # ---- 2d. Assemble final feature matrix ------------------------------------
    def build_user_feature_matrix(self) -> pd.DataFrame:
        """
        Joins all engineered feature blocks with static user attributes and
        the contest_rating label, returning the table predictor.py trains on.
        """
        features = (
            self.compute_accuracy_ratios()
            .join(self.compute_difficulty_distribution(), how="left")
            .join(self.compute_recency_momentum(), how="left")
            .fillna(0.0)
        )

        full = self.users_df.set_index("user_id").join(features, how="left").fillna(0.0)
        full = full.drop(columns=["latent_skill"])  # never leak the generative ground truth into modeling
        return full.reset_index()

    # ---- 2e. Raw failed submissions for the NLP engine ------------------------
    def get_failed_submissions_with_text(self, user_id: int | None = None) -> pd.DataFrame:
        """
        Returns failed submissions joined with question descriptions --
        exactly the payload nlp_cluster.py needs to embed and cluster.
        """
        sub = self.submissions_df
        failed = sub[sub["status"] != "Accepted"]
        if user_id is not None:
            failed = failed[failed["user_id"] == user_id]
        return failed.merge(
            self.questions_df[["question_id", "title", "description"]],
            on="question_id", how="left",
        )


if __name__ == "__main__":
    users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
    fe = FeatureEngineer(users_df, questions_df, submissions_df)
    feature_matrix = fe.build_user_feature_matrix()
    print(feature_matrix.head())
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
