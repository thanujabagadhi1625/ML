"""
data_processing.py
-------------------
DATA ENGINE

Responsibilities:
   1. Generate a realistic, internally-consistent synthetic dataset
      (questions, users, submissions) that mirrors a real LeetCode data
      export -- this is the drop-in replacement point for a real user export or synced submission log.
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

import ast
import json
import re
from pathlib import Path
from typing import Any

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


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_SEED if seed is None else seed)


def load_canonical_questions(n: int | None = None) -> pd.DataFrame:
    """
    Loads canonical questions from models/canonical_questions.json.
    Falls back to generating if missing.
    """
    path = config.CANONICAL_QUESTIONS_PATH
    if path.exists():
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(records)
            if n is not None and n < len(df):
                df = df.head(n)
            return df[QUESTION_COLUMNS]
        except Exception:
            pass
    # Fallback to template generation if canonical catalogue is unavailable
    return _generate_template_synthetic_questions(n or config.NUM_QUESTIONS)


def _generate_template_synthetic_questions(n: int = config.NUM_QUESTIONS) -> pd.DataFrame:
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
        templates[i].format(
            a=subjects[i], b=goals[i], c=", ".join(topic_tags_col[i]),
            d=difficulties[i].lower() + "-tier constraints"
        )
        for i in range(n)
    ]
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


def generate_synthetic_questions(n: int = config.NUM_QUESTIONS) -> pd.DataFrame:
    """
    Returns canonical questions matching the question universe, with length n.
    """
    return load_canonical_questions(n)


def generate_synthetic_users(
    n: int = config.NUM_USERS,
    random_seed: int = config.RANDOM_SEED,
    return_latent_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a synthetic `users` table driven by realistic user archetypes.

    Simulation Design (Latent Simulation Variables vs Observable Features):
    - LATENT SIMULATION VARIABLES (ground truth used only for event generation):
        * archetype: Behavioral archetype name (e.g. weak_dp, improving, specialized)
        * base_skill: Ground-truth baseline ability
        * topic_affinities: Dict of topic-specific skill offsets
        * hard_resilience: Specific ability offset when facing Hard problems
        * learning_rate_trend: Temporal progress slope over 365 days
        * attempt_tendency: Tendency to make multiple rapid attempts per problem
    - OBSERVABLE MODEL FEATURES (rediscovered by ML models via submission logs):
        * overall_accuracy, accuracy_easy, accuracy_medium, accuracy_hard
        * share_easy, share_medium, share_hard, recency_momentum, recent_submission_count,
          attempts_per_problem, failure_rate, account_age_days
    - TARGET (regression label for XGBoost rating model):
        * contest_rating: Modeled as a function of the user's final true skill and experience.

    Latent simulation variables are strictly NEVER passed as input features to predictive models!
    """
    rng = _rng(random_seed)
    archetype_names = config.USER_ARCHETYPES
    archetypes = [archetype_names[i % len(archetype_names)] for i in range(n)]

    base_skills = np.zeros(n)
    hard_resilience = np.zeros(n)
    learning_slopes = np.zeros(n)
    attempt_multipliers = np.ones(n)
    topic_affinities_list = []

    for i, arch in enumerate(archetypes):
        aff = {}
        if arch == "strong_overall":
            base_skills[i] = rng.normal(1.6, 0.25)
            hard_resilience[i] = 0.5
        elif arch == "weak_dp":
            base_skills[i] = rng.normal(0.8, 0.25)
            aff["Dynamic Programming"] = -2.0
        elif arch == "weak_graph":
            base_skills[i] = rng.normal(0.8, 0.25)
            for t in ["Graph", "Tree", "Depth-First Search", "Breadth-First Search"]:
                aff[t] = -1.8
        elif arch == "strong_easy_med_weak_hard":
            base_skills[i] = rng.normal(1.1, 0.25)
            hard_resilience[i] = -2.2
        elif arch == "improving":
            base_skills[i] = -1.2
            learning_slopes[i] = 2.4 / 365.0  # begins at -1.2, reaches +1.2 by day 365
        elif arch == "declining":
            base_skills[i] = 1.2
            learning_slopes[i] = -2.4 / 365.0  # begins at +1.2, drops to -1.2 by day 365
        elif arch == "stable":
            base_skills[i] = rng.normal(0.0, 0.35)
        elif arch == "high_attempt_low_accuracy":
            base_skills[i] = rng.normal(-0.6, 0.25)
            attempt_multipliers[i] = 2.5
        elif arch == "specialized":
            base_skills[i] = rng.normal(0.2, 0.25)
            for t in ["Array", "String", "Math", "Two Pointers"]:
                aff[t] = 1.5
            for t in ["Dynamic Programming", "Graph", "Tree", "Trie", "Backtracking"]:
                aff[t] = -1.5

        topic_affinities_list.append(aff)

    account_age_days = rng.integers(60, 1400, size=n)
    # The user's final effective skill at the evaluation horizon (day 365)
    final_skill = base_skills + learning_slopes * np.minimum(account_age_days, 365)

    # Ground-truth synthetic contest rating
    contest_rating = (
        1200.0
        + 260.0 * final_skill
        + 45.0 * hard_resilience
        + 0.08 * np.sqrt(account_age_days) * 10.0
        + rng.normal(0, 35.0, size=n)
    )
    contest_rating = np.clip(contest_rating, 800.0, 3000.0)

    users_df = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "account_age_days": account_age_days,
        "latent_skill": final_skill,
        "contest_rating": contest_rating,
    })

    if return_latent_info:
        latent_df = pd.DataFrame({
            "user_id": np.arange(1, n + 1),
            "archetype": archetypes,
            "base_skill": base_skills,
            "hard_resilience": hard_resilience,
            "learning_slope": learning_slopes,
            "attempt_multiplier": attempt_multipliers,
            "topic_affinities": topic_affinities_list,
        })
        return users_df[USER_COLUMNS], latent_df

    return users_df[USER_COLUMNS]


def generate_synthetic_submissions(
    users_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    n_submissions: int = config.NUM_SUBMISSIONS,
    random_seed: int = config.RANDOM_SEED,
    latent_users_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Builds a synthetic submissions log reflecting user archetypes, item response theory,
    topic affinities, temporal progress, and bursty re-attempts.
    """
    rng = _rng(random_seed)
    n_users = len(users_df)
    n_questions = len(questions_df)

    user_ids = users_df["user_id"].to_numpy()
    question_ids = questions_df["question_id"].to_numpy()
    question_diffs = questions_df["difficulty"].map(config.DIFFICULTY_SCORE).to_numpy()

    # Pre-map question topics
    tag_lookup = dict(zip(questions_df["question_id"], questions_df["topic_tags"]))
    diff_lookup = dict(zip(questions_df["question_id"], question_diffs))

    # Retrieve or reconstruct latent archetype profiles
    if latent_users_info is not None:
        latent_info = latent_users_info.set_index("user_id").to_dict(orient="index")
    else:
        # Reconstruct archetype parameters deterministically from user index
        latent_info = {}
        archetype_names = config.USER_ARCHETYPES
        for u in user_ids:
            arch = archetype_names[(u - 1) % len(archetype_names)]
            base_skill = 0.0
            hard_res = 0.0
            slope = 0.0
            att_mult = 1.0
            aff = {}
            if arch == "strong_overall":
                base_skill = 1.6; hard_res = 0.5
            elif arch == "weak_dp":
                base_skill = 0.8; aff["Dynamic Programming"] = -2.0
            elif arch == "weak_graph":
                base_skill = 0.8
                for t in ["Graph", "Tree", "Depth-First Search", "Breadth-First Search"]: aff[t] = -1.8
            elif arch == "strong_easy_med_weak_hard":
                base_skill = 1.1; hard_res = -2.2
            elif arch == "improving":
                base_skill = -1.2; slope = 2.4 / 365.0
            elif arch == "declining":
                base_skill = 1.2; slope = -2.4 / 365.0
            elif arch == "stable":
                base_skill = 0.0
            elif arch == "high_attempt_low_accuracy":
                base_skill = -0.6; att_mult = 2.5
            elif arch == "specialized":
                base_skill = 0.2
                for t in ["Array", "String", "Math", "Two Pointers"]: aff[t] = 1.5
                for t in ["Dynamic Programming", "Graph", "Tree", "Trie", "Backtracking"]: aff[t] = -1.5

            latent_info[u] = {
                "archetype": arch,
                "base_skill": base_skill,
                "hard_resilience": hard_res,
                "learning_slope": slope,
                "attempt_multiplier": att_mult,
                "topic_affinities": aff,
            }

    # Weight user activity: high_attempt users submit more often
    user_weights = np.array([latent_info[u]["attempt_multiplier"] for u in user_ids])
    user_weights /= user_weights.sum()

    sub_user_ids = rng.choice(user_ids, size=n_submissions, p=user_weights)

    # Pre-build per-archetype problem selection distributions across canonical questions
    archetype_q_weights = {}
    for arch in config.USER_ARCHETYPES:
        w = np.ones(n_questions, dtype=float)
        for j, qid in enumerate(question_ids):
            tags_j = tag_lookup.get(qid, [])
            diff_j = questions_df.iloc[j]["difficulty"] if "difficulty" in questions_df.columns else "Medium"
            # Difficulty preference
            if arch == "strong_overall":
                w[j] *= 3.0 if diff_j == "Hard" else 1.2 if diff_j == "Medium" else 0.4
            elif arch in ("strong_easy_med_weak_hard", "high_attempt_low_accuracy"):
                w[j] *= 0.15 if diff_j == "Hard" else 1.0 if diff_j == "Medium" else 1.8

            # Topic focus
            if arch == "weak_dp" and "Dynamic Programming" in tags_j:
                w[j] *= 3.5
            elif arch == "weak_graph" and any(t in tags_j for t in ["Graph", "Tree", "Depth-First Search", "Breadth-First Search"]):
                w[j] *= 3.0
            elif arch == "specialized":
                if any(t in tags_j for t in ["Array", "String", "Math", "Two Pointers"]):
                    w[j] *= 2.5
                else:
                    w[j] *= 0.4
        archetype_q_weights[arch] = w / w.sum()

    sub_q_ids = np.zeros(n_submissions, dtype=int)
    for u in user_ids:
        mask = np.where(sub_user_ids == u)[0]
        cnt = len(mask)
        if cnt > 0:
            arch = latent_info[u]["archetype"]
            p_dist = archetype_q_weights[arch]
            p_retry = 0.50 if arch == "high_attempt_low_accuracy" else 0.20

            curr_q = rng.choice(question_ids, p=p_dist)
            u_qs = []
            for _ in range(cnt):
                if u_qs and rng.random() < p_retry:
                    u_qs.append(curr_q)
                else:
                    curr_q = rng.choice(question_ids, p=p_dist)
                    u_qs.append(curr_q)
            sub_q_ids[mask] = u_qs

    # Days ago: submissions span DATASET_WINDOW_DAYS with power-law recency skew
    days_ago = (rng.power(1.4, size=n_submissions) * config.DATASET_WINDOW_DAYS).astype(float)
    ref_date = getattr(config, "REFERENCE_DATE", pd.Timestamp("2026-01-01").normalize())
    timestamps = ref_date - pd.to_timedelta(days_ago, unit="D")

    # Vectorized / batched solve probability calculation
    solve_probs = np.zeros(n_submissions, dtype=float)
    for i in range(n_submissions):
        uid = sub_user_ids[i]
        qid = sub_q_ids[i]
        d_ago = days_ago[i]
        info = latent_info[uid]

        # Temporal skill at the moment of submission
        day_t = max(0.0, config.DATASET_WINDOW_DAYS - d_ago)
        current_skill = info["base_skill"] + info["learning_slope"] * day_t

        # Topic affinity
        q_tags = tag_lookup.get(qid, [])
        affs = info["topic_affinities"]
        topic_eff = 0.0
        if q_tags:
            matches = [affs[t] for t in q_tags if t in affs]
            if matches:
                topic_eff = float(np.mean(matches))

        q_diff = diff_lookup.get(qid, 2.2)
        # Apply hard resilience penalty if difficulty is Hard (diff >= 3.0)
        eff_diff = q_diff - (info["hard_resilience"] if q_diff >= 3.0 else 0.0)

        logit = 1.4 * (current_skill + topic_eff - (eff_diff - 2.2))
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))
        solve_probs[i] = prob

    is_accepted = rng.binomial(1, np.clip(solve_probs, 0.02, 0.98))

    fail_statuses = config.SUBMISSION_STATUSES[1:]
    fail_weights = config.STATUS_BASE_WEIGHTS[1:] / config.STATUS_BASE_WEIGHTS[1:].sum()
    fail_choice = rng.choice(fail_statuses, size=n_submissions, p=fail_weights)

    status = np.where(is_accepted == 1, "Accepted", fail_choice)
    runtime_ms = rng.gamma(shape=2.0, scale=45.0, size=n_submissions)
    language = rng.choice(config.LANGUAGES, size=n_submissions, p=[0.5, 0.2, 0.15, 0.1, 0.05])

    df = pd.DataFrame({
        "user_id": sub_user_ids,
        "question_id": sub_q_ids,
        "timestamp": timestamps,
        "status": status,
        "runtime_ms": runtime_ms,
        "language": language,
    }).sort_values("timestamp").reset_index(drop=True)

    return df[SUBMISSION_COLUMNS]


def user_train_val_test_split(
    users_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    train_ratio: float = config.TRAIN_USER_RATIO,
    val_ratio: float = config.VAL_USER_RATIO,
    test_ratio: float = config.TEST_USER_RATIO,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs clean USER-LEVEL partitioning into train, validation, and test sets.
    Guarantee: Submissions from the same user are NEVER split across sets, avoiding leakage.
    """
    rng = _rng(random_seed)
    uids = np.array(sorted(users_df["user_id"].unique()))
    rng.shuffle(uids)

    n_users = len(uids)
    n_train = int(round(train_ratio * n_users))
    n_val = int(round(val_ratio * n_users))

    train_uids = set(uids[:n_train])
    val_uids = set(uids[n_train:n_train + n_val])
    test_uids = set(uids[n_train + n_val:])

    train_users = users_df[users_df["user_id"].isin(train_uids)].reset_index(drop=True)
    val_users = users_df[users_df["user_id"].isin(val_uids)].reset_index(drop=True)
    test_users = users_df[users_df["user_id"].isin(test_uids)].reset_index(drop=True)

    train_subs = submissions_df[submissions_df["user_id"].isin(train_uids)].reset_index(drop=True)
    val_subs = submissions_df[submissions_df["user_id"].isin(val_uids)].reset_index(drop=True)
    test_subs = submissions_df[submissions_df["user_id"].isin(test_uids)].reset_index(drop=True)

    return train_users, val_users, test_users, train_subs, val_subs, test_subs


def temporal_split_user_submissions(
    submissions_df: pd.DataFrame,
    history_ratio: float = config.TEMPORAL_HISTORY_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs chronological splitting within each user's submissions:
    First `history_ratio` fraction as historical context, remainder as held-out future interactions.
    Used for temporal evaluation of recommendations and weak-topic detection.
    """
    sub_sorted = submissions_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    history_rows = []
    future_rows = []

    for uid, group in sub_sorted.groupby("user_id"):
        n_items = len(group)
        if n_items <= 2:
            history_rows.append(group)
            continue
        split_idx = max(1, int(round(n_items * history_ratio)))
        split_idx = min(split_idx, n_items - 1)
        history_rows.append(group.iloc[:split_idx])
        future_rows.append(group.iloc[split_idx:])

    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else sub_sorted.iloc[:0]
    future_df = pd.concat(future_rows, ignore_index=True) if future_rows else sub_sorted.iloc[:0]
    return history_df, future_df


def generate_full_synthetic_dataset():
    """Convenience entry point returning (users_df, questions_df, submissions_df)."""
    questions_df = generate_synthetic_questions()
    users_df, latent_info = generate_synthetic_users(return_latent_info=True)
    submissions_df = generate_synthetic_submissions(users_df, questions_df, latent_users_info=latent_info)
    return users_df, questions_df, submissions_df

def _parse_topic_tags(value: Any) -> list[str]:
    raw_list = []
    if isinstance(value, list):
        raw_list = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                raw_list = [str(item).strip() for item in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed).strip()]
            except (ValueError, SyntaxError):
                raw_list = [tag.strip() for tag in text.split(",") if tag.strip()]
        elif text:
            raw_list = [tag.strip() for tag in text.split(",") if tag.strip()]

    clean_tags = []
    for tag in raw_list:
        t_clean = str(tag).strip()
        if t_clean in CANONICAL_LEETCODE_TOPICS:
            if t_clean not in clean_tags:
                clean_tags.append(t_clean)
        else:
            t_lower = t_clean.lower()
            if "dp" in t_lower or "dynamic" in t_lower:
                if "Dynamic Programming" not in clean_tags: clean_tags.append("Dynamic Programming")
            elif "graph" in t_lower:
                if "Graph" not in clean_tags: clean_tags.append("Graph")
            elif "tree" in t_lower:
                if "Tree" not in clean_tags: clean_tags.append("Tree")
            elif "string" in t_lower:
                if "String" not in clean_tags: clean_tags.append("String")
            elif "array" in t_lower:
                if "Array" not in clean_tags: clean_tags.append("Array")
    return clean_tags


CANONICAL_LEETCODE_TOPICS = {
    "Array", "String", "Hash Table", "Dynamic Programming", "Math",
    "Sorting", "Greedy", "Depth-First Search", "Breadth-First Search",
    "Binary Search", "Tree", "Binary Tree", "Binary Search Tree",
    "Graph", "Graph Traversal", "Two Pointers", "Sliding Window",
    "Backtracking", "Bit Manipulation", "Heap (Priority Queue)",
    "Stack", "Queue", "Monotonic Stack", "Monotonic Queue", "Trie",
    "Union Find", "Divide and Conquer", "Recursion", "Memoization",
    "Matrix", "Prefix Sum", "Counting", "Segment Tree",
    "Binary Indexed Tree", "Topological Sort", "Shortest Path",
    "Game Theory", "Combinatorics", "Geometry", "Linked List",
    "Doubly-Linked List", "Bitmask", "Simulation"
}

_KNOWN_PROBLEM_TAGS = {
    "two-sum": ["Array", "Hash Table"],
    "add-two-numbers": ["Linked List", "Math"],
    "longest-substring-without-repeating-characters": ["Hash Table", "String", "Sliding Window"],
    "median-of-two-sorted-arrays": ["Array", "Binary Search", "Divide and Conquer"],
    "longest-palindromic-substring": ["String", "Dynamic Programming"],
    "reverse-integer": ["Math"],
    "next-permutation": ["Array", "Two Pointers"],
    "combination-sum": ["Array", "Backtracking"],
    "combination-sum-ii": ["Array", "Backtracking"],
    "validate-binary-search-tree": ["Tree", "Depth-First Search", "Binary Search Tree"],
    "maximum-difference-between-node-and-ancestor": ["Tree", "Depth-First Search", "Breadth-First Search"],
    "time-needed-to-rearrange-a-binary-string": ["String", "Dynamic Programming", "Simulation"],
    "number-of-pairs-of-interchangeable-rectangles": ["Array", "Hash Table", "Math"],
    "minimized-maximum-of-products-distributed-to-any-store": ["Array", "Binary Search"],
    "minimum-operations-to-make-binary-array-elements-equal-to-one-ii": ["Array", "Greedy"],
}

_TAG_KEYWORD_RULES = [
    (r"\b(dp|dynamic|subsequence|knapsack|climbing-stairs|house-robber|stone-game|edit-distance|lis)\b", ["Dynamic Programming"]),
    (r"\b(bst|binary-search-tree)\b", ["Binary Search Tree", "Tree"]),
    (r"\b(tree|node|ancestor|root)\b", ["Tree", "Binary Tree"]),
    (r"\b(graph|network|components)\b", ["Graph"]),
    (r"\b(dfs|depth-first)\b", ["Depth-First Search"]),
    (r"\b(bfs|breadth-first)\b", ["Breadth-First Search"]),
    (r"\b(binary-search|closest-nodes|search-in|minimized-maximum)\b", ["Binary Search"]),
    (r"\b(two-pointers|two-sum|three-sum|container-with-most-water|valid-split)\b", ["Two Pointers", "Array"]),
    (r"\b(sliding-window)\b", ["Sliding Window", "String"]),
    (r"\b(substring|string|vowel|palindrome|characters|character|special-characters|suffix)\b", ["String"]),
    (r"\b(linked-list|reverse-integer|list-node|circular-game)\b", ["Linked List"]),
    (r"\b(stack|parentheses|validate-stack)\b", ["Stack"]),
    (r"\b(queue|circular-queue)\b", ["Queue"]),
    (r"\b(heap|priority-queue|kth-largest|closest-k)\b", ["Heap (Priority Queue)"]),
    (r"\b(union-find|disjoint)\b", ["Union Find"]),
    (r"\b(trie|prefix-tree)\b", ["Trie"]),
    (r"\b(bit|bits|binary-representation|flips|monobit|xor)\b", ["Bit Manipulation"]),
    (r"\b(backtracking|combination-sum|permutations|subsets|n-queens)\b", ["Backtracking", "Array"]),
    (r"\b(sort|sorting|sorted|rearrange)\b", ["Sorting"]),
    (r"\b(hash|map|pairs-of|interchangeable|dict)\b", ["Hash Table"]),
    (r"\b(grid|matrix|diagonal-traverse)\b", ["Matrix"]),
    (r"\b(prefix-sum)\b", ["Prefix Sum"]),
    (r"\b(greedy|minimum-operations|min-operations|maximum-difference)\b", ["Greedy"]),
    (r"\b(math|number-of|count-of|reduce-a-number|rectangle|integer|gcd|lcm)\b", ["Math"]),
    (r"\b(array|elements|subarray)\b", ["Array"]),
]


def _infer_topic_tags_from_slug_or_title(slug: Any, title: Any) -> list[str]:
    slug_str = str(slug).strip().lower() if slug else ""
    if slug_str in _KNOWN_PROBLEM_TAGS:
        return list(_KNOWN_PROBLEM_TAGS[slug_str])

    title_str = str(title).strip().lower() if title else ""
    for k, tags in _KNOWN_PROBLEM_TAGS.items():
        if k.replace("-", " ") in title_str or title_str.replace(" ", "-") == k:
            return list(tags)

    text = (slug_str + " " + title_str).lower()
    matched = []
    for pattern, tags in _TAG_KEYWORD_RULES:
        if re.search(pattern, text):
            for t in tags:
                if t in CANONICAL_LEETCODE_TOPICS and t not in matched:
                    matched.append(t)
    if not matched:
        if "str" in text or "word" in text or "char" in text:
            matched = ["String"]
        elif "num" in text or "sum" in text or "val" in text:
            matched = ["Math"]
        else:
            matched = ["Array"]
    return matched


def calculate_data_diagnostics(submissions_df: pd.DataFrame, questions_df: pd.DataFrame) -> dict:
    """Calculates internal diagnostics to catch bad/empty data before ML processing."""
    total = len(submissions_df)
    if total == 0:
        return {
            "total_submissions": 0,
            "unique_submissions": 0,
            "solved_submissions": 0,
            "failed_submissions": 0,
            "number_of_topics": 0,
            "date_range": {"earliest": None, "latest": None},
            "missing_fields": 0,
            "duplicate_count": 0,
        }

    solved = int((submissions_df["status"] == "Accepted").sum())
    failed = total - solved

    unique_subs = len(submissions_df.drop_duplicates(subset=["user_id", "question_id", "timestamp"]))
    dup_count = total - unique_subs

    all_topics = set()
    for tags in questions_df["topic_tags"]:
        if isinstance(tags, list):
            for t in tags:
                if t:
                    all_topics.add(t)

    min_ts = str(submissions_df["timestamp"].min()) if not submissions_df["timestamp"].isna().all() else None
    max_ts = str(submissions_df["timestamp"].max()) if not submissions_df["timestamp"].isna().all() else None

    missing = int(submissions_df[["question_id", "status", "timestamp"]].isna().sum().sum())

    return {
        "total_submissions": total,
        "unique_submissions": unique_subs,
        "solved_submissions": solved,
        "failed_submissions": failed,
        "number_of_topics": len(all_topics),
        "date_range": {"earliest": min_ts, "latest": max_ts},
        "missing_fields": missing,
        "duplicate_count": dup_count,
    }


def _load_export_dataframe(export_path: str | Path) -> pd.DataFrame:
    path = Path(export_path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            if "submissions" in payload:
                payload = payload["submissions"]
            elif "data" in payload and isinstance(payload["data"], dict):
                if "recentSubmissions" in payload["data"]:
                    payload = payload["data"]["recentSubmissions"]
                else:
                    raise ValueError("JSON export must contain a top-level 'submissions' list or a GraphQL-like data structure.")
            else:
                raise ValueError("JSON export must contain a top-level 'submissions' list or a GraphQL-like data structure.")

        if isinstance(payload, list):
            payload = pd.DataFrame(payload)
        else:
            raise ValueError("Unsupported JSON export format.")
    elif path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        payload = pd.read_csv(path, sep=sep)
    else:
        raise ValueError("Unsupported export format. Supported formats are JSON and CSV/TSV.")
    return payload


def _normalize_export_submissions(raw_df: pd.DataFrame) -> pd.DataFrame:
    sub = raw_df.copy()

    if "question_id" not in sub.columns:
        if "question_slug" in sub.columns:
            sub["question_id"] = sub["question_slug"]
        elif "problem_slug" in sub.columns:
            sub["question_id"] = sub["problem_slug"]
        elif "question_title" in sub.columns:
            sub["question_id"] = sub["question_title"]
        elif "problem_title" in sub.columns:
            sub["question_id"] = sub["problem_title"]
        elif "title" in sub.columns:
            sub["question_id"] = sub["title"]
        else:
            raise ValueError("Exported submissions must include question_id, question_slug, problem_slug, question_title, problem_title, or title.")

    if "title" not in sub.columns:
        if "problem_title" in sub.columns:
            sub["title"] = sub["problem_title"].astype(str)
        elif "question_title" in sub.columns:
            sub["title"] = sub["question_title"].astype(str)
        else:
            sub["title"] = sub["question_id"].astype(str)

    if "difficulty" not in sub.columns:
        if "question_difficulty" in sub.columns:
            sub["difficulty"] = sub["question_difficulty"]
        else:
            sub["difficulty"] = "Medium"

    if "status" not in sub.columns:
        if "result" in sub.columns:
            sub["status"] = sub["result"]
        elif "status_display" in sub.columns:
            sub["status"] = sub["status_display"]
        else:
            raise ValueError("Exported submissions must include a status column.")
    else:
        if "status_display" in sub.columns:
            sub["status"] = sub["status_display"].fillna(sub["status"])
        elif pd.api.types.is_numeric_dtype(sub["status"]):
            status_map = {
                10: "Accepted",
                11: "Wrong Answer",
                12: "Runtime Error",
                13: "Time Limit Exceeded",
                14: "Compile Error",
            }
            sub["status"] = sub["status"].replace(status_map).astype(str)

    if "timestamp" not in sub.columns:
        if "time" in sub.columns:
            sub["timestamp"] = sub["time"]
        elif "submission_time" in sub.columns:
            sub["timestamp"] = sub["submission_time"]
        else:
            raise ValueError("Exported submissions must include a timestamp column.")

    if "runtime_ms" not in sub.columns:
        if "runtime" in sub.columns:
            sub["runtime_ms"] = (
                sub["runtime"].astype(str).str.extract(r"(\d+)")[0].astype(float).fillna(0.0)
            )
        else:
            sub["runtime_ms"] = 0.0

    if "language" not in sub.columns:
        sub["language"] = "Python3"

    if "topic_tags" not in sub.columns:
        if "topics" in sub.columns:
            sub["topic_tags"] = sub["topics"]
        elif "tags" in sub.columns:
            sub["topic_tags"] = sub["tags"]
        else:
            sub["topic_tags"] = []

    sub["topic_tags"] = sub["topic_tags"].apply(_parse_topic_tags)

    def _fill_empty_tags(row):
        tags = row["topic_tags"]
        if not tags:
            slug = row.get("problem_slug") or row.get("question_slug") or row.get("question_id")
            title = row.get("title") or row.get("problem_title")
            return _infer_topic_tags_from_slug_or_title(slug, title)
        return tags

    sub["topic_tags"] = sub.apply(_fill_empty_tags, axis=1)

    if "description" not in sub.columns:
        title_text = sub["title"].astype(str)
        topic_text = sub["topic_tags"].astype(str)
        sub["description"] = (title_text + " " + sub["difficulty"].astype(str) + " " + topic_text).str.strip()

    if "acceptance_rate" not in sub.columns:
        sub["acceptance_rate"] = 0.5

    if "user_id" not in sub.columns:
        sub["user_id"] = 1

    if pd.api.types.is_numeric_dtype(sub["timestamp"]):
        numeric_ts = pd.to_numeric(sub["timestamp"], errors="coerce")
        if numeric_ts.notna().any():
            unit = "s" if numeric_ts.max() < 1e12 else "ms"
            sub["timestamp"] = pd.to_datetime(numeric_ts, unit=unit, errors="coerce")
        else:
            sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    else:
        as_num = pd.to_numeric(sub["timestamp"].astype(str).str.strip(), errors="coerce")
        if as_num.notna().any() and as_num.dropna().astype(int).astype(str).str.match(r"^\d{9,}$").all():
            unit = "s" if as_num.max() < 1e12 else "ms"
            sub["timestamp"] = pd.to_datetime(as_num, unit=unit, errors="coerce")
        else:
            sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")

    sub = sub.dropna(subset=["question_id", "status", "timestamp"])

    return sub[
        ["user_id", "question_id", "timestamp", "status", "runtime_ms", "language", "title", "difficulty", "topic_tags", "acceptance_rate"]
    ]


def _build_questions_from_export(submissions_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[Any, int]]:
    """
    Aligns exported submissions with the canonical question catalogue.
    Maps known problem slugs / titles to their canonical integer question_id.
    New/unseen problems receive deterministic new IDs beyond the canonical range.
    Returns the comprehensive questions_df (canonical universe + unseen user problems)
    and the id mapping for submissions_df.
    """
    canonical_df = load_canonical_questions()
    canonical_slug_to_id = {}
    canonical_title_to_id = {}
    if "slug" in canonical_df.columns:
        canonical_slug_to_id = dict(zip(canonical_df["slug"].astype(str).str.lower(), canonical_df["question_id"]))
    canonical_title_to_id = dict(zip(canonical_df["title"].astype(str).str.lower(), canonical_df["question_id"]))

    max_canonical_id = int(canonical_df["question_id"].max()) if not canonical_df.empty else 0

    unique_export = (
        submissions_df[["question_id", "title", "difficulty", "topic_tags", "acceptance_rate"]]
        .drop_duplicates(subset=["question_id"])
        .copy()
    )

    question_id_map: dict[Any, int] = {}
    extra_questions = []
    next_new_id = max_canonical_id + 1

    for _, row in unique_export.iterrows():
        raw_key = row["question_id"]
        key_str = str(raw_key).strip().lower()
        title_str = str(row["title"]).strip().lower()

        matched_id = None
        if key_str in canonical_slug_to_id:
            matched_id = canonical_slug_to_id[key_str]
        elif title_str in canonical_title_to_id:
            matched_id = canonical_title_to_id[title_str]
        elif key_str.isdigit() and int(key_str) in set(canonical_df["question_id"]):
            matched_id = int(key_str)

        if matched_id is not None:
            question_id_map[raw_key] = matched_id
        else:
            assigned_id = next_new_id
            next_new_id += 1
            question_id_map[raw_key] = assigned_id
            desc = f"{row['title']} {row['difficulty']} problem statement"
            extra_questions.append({
                "question_id": assigned_id,
                "title": row["title"],
                "description": desc,
                "difficulty": row["difficulty"],
                "topic_tags": row["topic_tags"],
                "acceptance_rate": row["acceptance_rate"],
            })

    if extra_questions:
        full_questions_df = pd.concat([canonical_df, pd.DataFrame(extra_questions)], ignore_index=True)
    else:
        full_questions_df = canonical_df.copy()

    return full_questions_df[QUESTION_COLUMNS], question_id_map


def _build_users_from_export(submissions_df: pd.DataFrame) -> pd.DataFrame:
    unique_users = submissions_df["user_id"].unique().tolist()
    rows = []
    for user_id in unique_users:
        rows.append({
            "user_id": user_id,
            "account_age_days": 365,
            "latent_skill": 0.0,
            "contest_rating": 1500.0,
        })
    return pd.DataFrame(rows)[USER_COLUMNS]


def load_leetcode_history_export(export_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _load_export_dataframe(export_path)
    submissions_df = _normalize_export_submissions(raw)
    questions_df, question_id_map = _build_questions_from_export(submissions_df)
    submissions_df["question_id"] = submissions_df["question_id"].map(question_id_map)

    unique_user_ids = submissions_df["user_id"].unique().tolist()
    user_id_map = {old_id: idx + 1 for idx, old_id in enumerate(sorted(unique_user_ids))}
    submissions_df["user_id"] = submissions_df["user_id"].map(user_id_map)
    users_df = _build_users_from_export(submissions_df)
    users_df["user_id"] = users_df["user_id"].map(user_id_map)

    return users_df[USER_COLUMNS], questions_df[QUESTION_COLUMNS], submissions_df[SUBMISSION_COLUMNS]


def load_leetcode_history_records(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame(records)
    if "user_id" not in raw_df.columns or raw_df["user_id"].isna().all():
        raw_df["user_id"] = 1
    raw_df["user_id"] = raw_df["user_id"].fillna(1).astype(int)

    submissions_df = _normalize_export_submissions(raw_df)
    questions_df, question_id_map = _build_questions_from_export(submissions_df)
    submissions_df["question_id"] = submissions_df["question_id"].map(question_id_map)

    unique_user_ids = submissions_df["user_id"].unique().tolist()
    user_id_map = {old_id: idx + 1 for idx, old_id in enumerate(sorted(unique_user_ids))}
    submissions_df["user_id"] = submissions_df["user_id"].map(user_id_map)
    users_df = _build_users_from_export(submissions_df)
    users_df["user_id"] = users_df["user_id"].map(user_id_map)

    return users_df[USER_COLUMNS], questions_df[QUESTION_COLUMNS], submissions_df[SUBMISSION_COLUMNS]


def generate_synthetic_dataset_for_questions(
    questions_df: pd.DataFrame,
    n_users: int = config.NUM_USERS,
    n_submissions: int = config.NUM_SUBMISSIONS,
    user_id_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users_df = generate_synthetic_users(n_users)
    submissions_df = generate_synthetic_submissions(users_df, questions_df, n_submissions)
    if user_id_offset:
        users_df = users_df.assign(user_id=users_df["user_id"] + user_id_offset)
        submissions_df = submissions_df.assign(user_id=submissions_df["user_id"] + user_id_offset)
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
        # Drop columns from submissions_df that will be merged from questions_df to avoid _x/_y suffix collision
        overlap_cols = [c for c in ["difficulty", "topic_tags", "acceptance_rate", "title", "description"] if c in submissions_df.columns]
        cleaned_submissions = submissions_df.drop(columns=overlap_cols) if overlap_cols else submissions_df
        self.submissions_df = cleaned_submissions.merge(
            questions_df[["question_id", "difficulty", "topic_tags", "acceptance_rate"]],
            on="question_id", how="left",
        )

    # ---- 2a. Accuracy ratios -------------------------------------------------
    def compute_accuracy_ratios(self) -> pd.DataFrame:
        """Overall and per-difficulty accuracy ratio, one row per user."""
        sub = self.submissions_df
        cols = ["total_submissions", "total_accepted", "overall_accuracy", "accuracy_easy", "accuracy_medium", "accuracy_hard"]
        if sub.empty:
            return pd.DataFrame(columns=cols, index=pd.Index([], name="user_id"))

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
        for col in ["accuracy_easy", "accuracy_medium", "accuracy_hard"]:
            if col not in per_difficulty.columns:
                per_difficulty[col] = 0.0

        return overall.join(per_difficulty, how="left").fillna(0.0)

    # ---- 2b. Difficulty distribution -----------------------------------------
    def compute_difficulty_distribution(self) -> pd.DataFrame:
        """Share of a user's total attempts spent on Easy/Medium/Hard problems."""
        sub = self.submissions_df
        cols = ["share_easy", "share_medium", "share_hard"]
        if sub.empty:
            return pd.DataFrame(columns=cols, index=pd.Index([], name="user_id"))

        counts = pd.pivot_table(
            sub, index="user_id", columns="difficulty", values="question_id",
            aggfunc="count", fill_value=0,
        )
        shares = counts.div(counts.sum(axis=1), axis=0).add_prefix("share_").rename(columns=lambda c: c.lower())
        for col in ["share_easy", "share_medium", "share_hard"]:
            if col not in shares.columns:
                shares[col] = 0.0
        return shares

    # ---- 2c. Recency / momentum ----------------------------------------------
    def compute_recency_momentum(self, half_life_days: float = config.RECENCY_HALF_LIFE_DAYS) -> pd.DataFrame:
        """
        Exponentially time-decayed "momentum" score per user:

            momentum = sum(weight_i * is_accepted_i) / sum(weight_i)
            lambda = ln(2) / half_life_days
            weight_i = exp(-lambda * days_ago_i)

        This rewards users who are *currently* solving well over users whose
        accuracy was good months ago but has since stalled -- much stronger
        signal for predicting a near-term contest rating than a flat average.
        """
        sub = self.submissions_df.copy()
        cols = ["recency_momentum", "recent_submission_count"]
        if sub.empty:
            return pd.DataFrame(columns=cols, index=pd.Index([], name="user_id"))

        now = sub["timestamp"].max()
        days_ago = (now - sub["timestamp"]).dt.total_seconds() / 86400.0
        decay_lambda = np.log(2.0) / half_life_days
        weight = np.exp(-decay_lambda * days_ago)

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

    # ---- 2d. Submission intensity & failure metrics --------------------------
    def compute_submission_intensity(self) -> pd.DataFrame:
        """Unique problem coverage, attempts per problem, and failure rate per user."""
        sub = self.submissions_df
        cols = ["unique_problems_attempted", "attempts_per_problem", "failure_rate"]
        if sub.empty:
            return pd.DataFrame(columns=cols, index=pd.Index([], name="user_id"))

        intensity = (
            sub.groupby("user_id")
            .agg(
                unique_problems_attempted=("question_id", "nunique"),
                total_attempts=("question_id", "count"),
                failed_attempts=("status", lambda s: (s != "Accepted").sum()),
            )
            .assign(
                attempts_per_problem=lambda d: d["total_attempts"] / np.maximum(d["unique_problems_attempted"], 1),
                failure_rate=lambda d: d["failed_attempts"] / np.maximum(d["total_attempts"], 1),
            )
            [["unique_problems_attempted", "attempts_per_problem", "failure_rate"]]
        )
        return intensity

    # ---- 2e. Assemble final feature matrix ------------------------------------
    def build_user_feature_matrix(self) -> pd.DataFrame:
        """
        Joins all engineered feature blocks with static user attributes and
        the contest_rating label, returning the table predictor.py trains on.
        Strictly drops all latent simulation variables to prevent target leakage.
        """
        features = (
            self.compute_accuracy_ratios()
            .join(self.compute_difficulty_distribution(), how="left")
            .join(self.compute_recency_momentum(), how="left")
            .join(self.compute_submission_intensity(), how="left")
            .fillna(0.0)
        )

        full = self.users_df.set_index("user_id").join(features, how="left").fillna(0.0)
        drop_leakage = [
            "latent_skill", "archetype", "base_skill", "hard_resilience",
            "learning_slope", "attempt_multiplier", "topic_affinities"
        ]
        full = full.drop(columns=[c for c in drop_leakage if c in full.columns], errors="ignore")
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

    def get_topic_profile(self, user_id: int = 1) -> pd.DataFrame:
        """Compute topic weakness profile for a specific user."""
        return compute_user_topic_profile(self.submissions_df, self.questions_df, user_id=user_id)


def compute_user_topic_profile(
    submissions_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    user_id: int = 1
) -> pd.DataFrame:
    """
    Computes empirical topic weakness profile for a specific user using actual submission history
    and LeetCode structured tags.
    """
    sub = submissions_df[submissions_df["user_id"] == user_id].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "topic", "attempts", "success_rate_pct", "success_rate_num", "exposure_level",
            "dominant_difficulty", "weakness_level", "risk_score", "unique_attempted", "unique_solved", "why_weak"
        ])

    if "topic_tags" not in sub.columns or "difficulty" not in sub.columns or sub["topic_tags"].isna().all():
        sub = sub.drop(columns=["topic_tags", "difficulty"], errors="ignore").merge(
            questions_df[["question_id", "difficulty", "topic_tags"]], on="question_id", how="left"
        )

    sub_tags = sub.explode("topic_tags").dropna(subset=["topic_tags"]).copy()
    sub_tags["topic_tags"] = sub_tags["topic_tags"].astype(str).str.strip()
    sub_tags = sub_tags[sub_tags["topic_tags"] != ""]

    if sub_tags.empty:
        return pd.DataFrame()

    sub_tags["is_accepted"] = (sub_tags["status"] == "Accepted").astype(int)

    # Time-decay weight for recency (21-day half life)
    if "timestamp" in sub_tags.columns and pd.api.types.is_datetime64_any_dtype(sub_tags["timestamp"]):
        now = sub_tags["timestamp"].max()
        days_ago = (now - sub_tags["timestamp"]).dt.total_seconds() / 86400.0
        sub_tags["_weight"] = np.power(0.5, days_ago / 21.0)
    else:
        sub_tags["_weight"] = 1.0

    sub_tags["_weighted_correct"] = sub_tags["_weight"] * sub_tags["is_accepted"]

    topic_records = []
    for topic_name, group in sub_tags.groupby("topic_tags"):
        attempts = len(group)
        accepted_cnt = int(group["is_accepted"].sum())
        failed_cnt = attempts - accepted_cnt
        success_rate = accepted_cnt / attempts if attempts > 0 else 0.0

        unique_attempted = int(group["question_id"].nunique())
        unique_solved = int(group[group["is_accepted"] == 1]["question_id"].nunique())

        # Exposure: how many distinct problems in topic attempted
        if unique_attempted < 5:
            exposure_level = "Low"
        elif unique_attempted <= 15:
            exposure_level = "Medium"
        else:
            exposure_level = "High"

        # Difficulty distribution & dominant difficulty
        diff_counts = group["difficulty"].value_counts()
        dominant_diff = diff_counts.index[0] if not diff_counts.empty else "Medium"

        # Difficulty weighting (failing Easy carries 1.3x risk, Hard 0.8x)
        easy_group = group[group["difficulty"] == "Easy"]
        med_group = group[group["difficulty"] == "Medium"]
        hard_group = group[group["difficulty"] == "Hard"]

        easy_fail_rate = 1.0 - (easy_group["is_accepted"].mean() if not easy_group.empty else 1.0)
        med_fail_rate = 1.0 - (med_group["is_accepted"].mean() if not med_group.empty else 1.0)

        # Recency momentum in topic
        recency_success = group["_weighted_correct"].sum() / group["_weight"].sum() if group["_weight"].sum() > 0 else success_rate

        # Weakness Risk Score & Level Classification with Cold-Start Guard (attempts < 3)
        if attempts < 3:
            weakness_level = "Neutral"
            risk_score = 0.0
        else:
            # Skill & Risk Composite Score:
            # 0.50 * (1 - Accuracy_decayed) + 0.30 * (1 - Accuracy_lifetime) + 0.20 * min(1, failures / 5)
            failures = max(0, attempts - accepted_cnt)
            risk_score = (
                0.50 * (1.0 - recency_success)
                + 0.30 * (1.0 - success_rate)
                + 0.20 * min(1.0, failures / 5.0)
            )

            if (success_rate <= 0.42 and attempts >= 8) or (risk_score >= 0.55 and attempts >= 5):
                weakness_level = "Critical"
            elif (success_rate <= 0.52 and attempts >= 5) or (risk_score >= 0.45 and attempts >= 4):
                weakness_level = "Weak"
            elif success_rate <= 0.68:
                weakness_level = "Moderate"
            else:
                weakness_level = "Strong"

        success_pct = f"{int(round(success_rate * 100))}%"

        why_weak = []
        if weakness_level in ["Critical", "Weak"]:
            why_weak.append(f"{attempts} attempts across {unique_attempted} unique problems (Exposure: {exposure_level})")
            why_weak.append(f"{success_pct} overall success rate ({accepted_cnt} solved, {failed_cnt} failed)")
            if recency_success < success_rate:
                why_weak.append(f"Recent performance ({int(round(recency_success * 100))}% success) shows declining momentum")
            if not easy_group.empty and easy_fail_rate > 0.3:
                why_weak.append(f"Easy-difficulty accuracy ({int(round((1.0 - easy_fail_rate) * 100))}% success) indicates fundamental pattern gaps")

        topic_records.append({
            "topic": topic_name,
            "attempts": attempts,
            "success_rate_pct": success_pct,
            "success_rate_num": success_rate,
            "exposure_level": exposure_level,
            "dominant_difficulty": dominant_diff,
            "weakness_level": weakness_level,
            "risk_score": risk_score,
            "unique_attempted": unique_attempted,
            "unique_solved": unique_solved,
            "why_weak": why_weak,
        })

    df_profile = pd.DataFrame(topic_records)
    level_order = {"Critical": 0, "Weak": 1, "Moderate": 2, "Strong": 3, "Neutral": 4}
    df_profile["sort_order"] = df_profile["weakness_level"].map(level_order)
    df_profile = df_profile.sort_values(["sort_order", "attempts"], ascending=[True, False]).drop(columns=["sort_order"])

    return df_profile


# ==============================================================================
# 1b. SYNTHETIC DATA GENERATOR V3 (Honest, Multi-Factor Generative Simulator)
# ==============================================================================
"""
Generative Simulator v3 Assumptions & Architecture:
--------------------------------------------------
1. Dirichlet Topic Affinity: Each user u is assigned a 20-dimensional latent topic interest vector
   theta_u ~ Dirichlet(alpha_arch). Archetypes shift the mean concentration vector alpha_arch,
   ensuring users within an archetype have correlated preferences while maintaining individual variance.
   Users of the same archetype do NOT share identical distributions.
2. Stochastic Individual Skill & Evolution: Base ability skill_u(0) ~ Normal(mu_arch, sigma_skill) has
   individual noise per user. Temporal progress skill_u(t) = skill_u(0) + slope_arch * elapsed_days
   evolves chronologically according to the archetype profile (improving, declining, stable).
3. Universal Long-Tailed Question Popularity: Problem base popularity follows a long-tailed Zipf/power-law
   distribution (fixed by dataset seed), reflecting realistic platform selection concentration.
4. Problem Choice Mechanics: The probability of user u selecting problem q at time t is proportional to:
   P(q | u, t) proportional to popularity(q) * exp(beta * (theta_u . tags_q)) * difficulty_fit(skill_u(t), diff_q)
   where difficulty_fit favors problems slightly above the user's current ability (skill + delta).
   Parameter beta scales topic preference strength.
5. Re-attempt Dynamics: After a failed submission, the user retries the same question with configurable
   probability p_retry; after an Accepted submission, the user transitions to a new problem choice.
6. IRT Solve Probability: Solve probability follows an Item Response Theory logistic model driven by
   the exact same time-varying skill skill_u(t) and topic match used during question selection.
7. Heavy-Tailed User Activity Volume: The number of submissions per user M_u follows a lognormal
   distribution with configurable mean (~100) and minimum 20.
8. Chronological Sequence & Consistent Rating Target: Submissions are generated in strictly increasing
   chronological order over span min(account_age_days, 365). The user's contest_rating label is
   computed from true skill at the END of this actual submission span, resolving historical duration mismatches.
"""


def generate_synthetic_users_v3(
    n: int = 2000,
    random_seed: int = config.RANDOM_SEED,
    return_latent_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    rng = _rng(random_seed)
    archetype_names = config.USER_ARCHETYPES
    archetypes = [archetype_names[i % len(archetype_names)] for i in range(n)]

    tag_to_idx = {t: i for i, t in enumerate(config.TOPIC_TAGS)}
    n_tags = len(config.TOPIC_TAGS)

    # Base concentration per archetype for Dirichlet topic interest
    archetype_alphas = {}
    for arch in archetype_names:
        a = np.ones(n_tags, dtype=float)
        if arch == "weak_dp":
            if "Dynamic Programming" in tag_to_idx:
                a[tag_to_idx["Dynamic Programming"]] = 8.0
        elif arch == "weak_graph":
            for t in ["Graph", "Tree", "Depth-First Search", "Breadth-First Search"]:
                if t in tag_to_idx:
                    a[tag_to_idx[t]] = 6.0
        elif arch == "specialized":
            for t in ["Array", "String", "Math", "Two Pointers"]:
                if t in tag_to_idx:
                    a[tag_to_idx[t]] = 6.0
        elif arch == "strong_overall":
            a[:] = 2.0
        archetype_alphas[arch] = a

    base_skills = np.zeros(n)
    hard_resilience = np.zeros(n)
    learning_slopes = np.zeros(n)
    topic_interests = np.zeros((n, n_tags), dtype=float)

    for i, arch in enumerate(archetypes):
        alpha = archetype_alphas[arch]
        topic_interests[i] = rng.dirichlet(alpha)

        # Individual skill noise
        s_noise = rng.normal(0.0, 0.20)
        h_noise = rng.normal(0.0, 0.10)

        if arch == "strong_overall":
            base_skills[i] = 1.6 + s_noise
            hard_resilience[i] = 0.5 + h_noise
        elif arch == "weak_dp":
            base_skills[i] = 0.8 + s_noise
            hard_resilience[i] = 0.0 + h_noise
        elif arch == "weak_graph":
            base_skills[i] = 0.8 + s_noise
            hard_resilience[i] = 0.0 + h_noise
        elif arch == "strong_easy_med_weak_hard":
            base_skills[i] = 1.1 + s_noise
            hard_resilience[i] = -2.2 + h_noise
        elif arch == "improving":
            base_skills[i] = -1.2 + s_noise
            learning_slopes[i] = 2.4 / 365.0
            hard_resilience[i] = 0.0 + h_noise
        elif arch == "declining":
            base_skills[i] = 1.2 + s_noise
            learning_slopes[i] = -2.4 / 365.0
            hard_resilience[i] = 0.0 + h_noise
        elif arch == "stable":
            base_skills[i] = 0.0 + rng.normal(0.0, 0.30)
            hard_resilience[i] = 0.0 + h_noise
        elif arch == "high_attempt_low_accuracy":
            base_skills[i] = -0.6 + s_noise
            hard_resilience[i] = -0.5 + h_noise
        elif arch == "specialized":
            base_skills[i] = 0.2 + s_noise
            hard_resilience[i] = 0.0 + h_noise

    account_age_days = rng.integers(60, 1400, size=n)
    submission_span = np.minimum(account_age_days, 365)
    final_skill = base_skills + learning_slopes * submission_span

    contest_rating = (
        1200.0
        + 260.0 * final_skill
        + 45.0 * hard_resilience
        + 0.08 * np.sqrt(submission_span) * 10.0
        + rng.normal(0, 35.0, size=n)
    )
    contest_rating = np.clip(contest_rating, 800.0, 3000.0)

    users_df = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "account_age_days": account_age_days,
        "latent_skill": final_skill,
        "contest_rating": contest_rating,
    })

    if return_latent_info:
        latent_df = pd.DataFrame({
            "user_id": np.arange(1, n + 1),
            "archetype": archetypes,
            "base_skill": base_skills,
            "hard_resilience": hard_resilience,
            "learning_slope": learning_slopes,
            "submission_span": submission_span,
            "topic_interest": [topic_interests[i] for i in range(n)],
        })
        return users_df[USER_COLUMNS], latent_df

    return users_df[USER_COLUMNS]


def generate_synthetic_submissions_v3(
    users_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    random_seed: int = config.RANDOM_SEED,
    latent_users_info: pd.DataFrame | None = None,
    beta: float = 1.5,
    p_retry: float = 0.35,
    mean_submissions: float = 100.0,
    min_submissions: int = 20,
    lognormal_sigma: float = 0.55,
    ref_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rng = _rng(random_seed)
    n_users = len(users_df)
    n_questions = len(questions_df)
    question_ids = questions_df["question_id"].to_numpy()

    if ref_date is None:
        ref_date = getattr(config, "REFERENCE_DATE", pd.Timestamp("2026-01-01").normalize())

    # Pre-map question difficulties and tags
    diff_map = config.DIFFICULTY_SCORE
    d_q = questions_df["difficulty"].map(diff_map).fillna(2.2).to_numpy()

    easy_mask = (d_q <= 1.5)
    med_mask = (d_q > 1.5) & (d_q < 3.0)
    hard_mask = (d_q >= 3.0)

    tag_to_idx = {t: i for i, t in enumerate(config.TOPIC_TAGS)}
    n_tags = len(config.TOPIC_TAGS)
    T_tags = np.zeros((n_questions, n_tags), dtype=float)
    for j, tags in enumerate(questions_df["topic_tags"]):
        for t in tags:
            if t in tag_to_idx:
                T_tags[j, tag_to_idx[t]] = 1.0

    # Universal long-tailed question popularity (Zipf distribution) fixed by seed
    pop_rng = np.random.default_rng(random_seed + 777)
    ranks = np.arange(1, n_questions + 1)
    pop_weights = 1.0 / (ranks ** 0.8)
    pop_rng.shuffle(pop_weights)
    pop_weights /= pop_weights.sum()

    # Latent user info
    if latent_users_info is not None:
        latent_dict = latent_users_info.set_index("user_id").to_dict(orient="index")
    else:
        _, lat_df = generate_synthetic_users_v3(n=n_users, random_seed=random_seed, return_latent_info=True)
        latent_dict = lat_df.set_index("user_id").to_dict(orient="index")

    # Lognormal heavy-tailed submissions per user
    mu = np.log(mean_submissions) - 0.5 * (lognormal_sigma ** 2)
    user_n_subs = np.maximum(
        min_submissions,
        np.round(rng.lognormal(mean=mu, sigma=lognormal_sigma, size=n_users)),
    ).astype(int)

    all_user_ids = []
    all_q_ids = []
    all_timestamps = []
    all_statuses = []
    all_runtimes = []
    all_languages = []

    fail_statuses = config.SUBMISSION_STATUSES[1:]
    fail_weights = config.STATUS_BASE_WEIGHTS[1:] / config.STATUS_BASE_WEIGHTS[1:].sum()
    lang_choices = config.LANGUAGES
    lang_weights = [0.5, 0.2, 0.15, 0.1, 0.05]

    for u_idx, row in users_df.reset_index(drop=True).iterrows():
        uid = int(row["user_id"])
        u_info = latent_dict[uid]
        M_u = int(user_n_subs[u_idx])
        T_u = float(min(row["account_age_days"], 365.0))
        base_skill = float(u_info["base_skill"])
        slope = float(u_info["learning_slope"])
        hard_res = float(u_info["hard_resilience"])
        theta_u = np.asarray(u_info["topic_interest"], dtype=float)

        # Base question selection mass for this user (popularity * exp(beta * theta . tags))
        tag_match = T_tags @ theta_u  # (N,)
        base_q_mass = pop_weights * np.exp(beta * tag_match)

        # Strictly increasing timestamps spanning min(age, 365)
        start_date = ref_date - pd.to_timedelta(T_u, unit="D")
        deltas = rng.exponential(scale=1.0, size=M_u) + 1e-4
        cum_deltas = np.cumsum(deltas)
        span_sec = max(T_u, 1.0) * 86400.0
        rel_sec = (cum_deltas / cum_deltas[-1]) * span_sec
        u_ts = start_date + pd.to_timedelta(rel_sec, unit="s")

        prev_qid = -1
        prev_j = -1
        prev_status = None

        for k in range(M_u):
            elapsed_days = rel_sec[k] / 86400.0
            current_skill = base_skill + slope * elapsed_days

            # Question choice
            if k > 0 and prev_status != "Accepted" and rng.random() < p_retry:
                q_k = prev_qid
                j_k = prev_j
            else:
                # Difficulty fit favors problems slightly above user's current skill
                s_star = current_skill + 0.35
                me = np.exp(-((-1.2 - s_star) ** 2) / 1.28)
                mm = np.exp(-((0.0 - s_star) ** 2) / 1.28)
                mh = np.exp(-((1.4 - s_star) ** 2) / 1.28)
                W_k = base_q_mass * np.where(easy_mask, me, np.where(med_mask, mm, mh))
                W_sum = W_k.sum()
                if W_sum <= 0:
                    p_k = np.ones(n_questions) / n_questions
                else:
                    p_k = W_k / W_sum
                cdf = np.cumsum(p_k)
                j_k = int(np.searchsorted(cdf, rng.random()))
                if j_k >= n_questions:
                    j_k = n_questions - 1
                q_k = question_ids[j_k]

            # Solve probability (IRT formula)
            q_diff = d_q[j_k]
            eff_diff = q_diff - (hard_res if q_diff >= 3.0 else 0.0)
            topic_eff = (tag_match[j_k] - 0.05) * 2.0
            logit = 1.4 * (current_skill + topic_eff - (eff_diff - 2.2))
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))
            prob = np.clip(prob, 0.02, 0.98)

            is_accepted = (rng.random() < prob)
            status_k = "Accepted" if is_accepted else rng.choice(fail_statuses, p=fail_weights)

            all_user_ids.append(uid)
            all_q_ids.append(q_k)
            all_timestamps.append(u_ts[k])
            all_statuses.append(status_k)
            all_runtimes.append(rng.gamma(shape=2.0, scale=45.0))
            all_languages.append(rng.choice(lang_choices, p=lang_weights))

            prev_qid = q_k
            prev_j = j_k
            prev_status = status_k

    df = pd.DataFrame({
        "user_id": all_user_ids,
        "question_id": all_q_ids,
        "timestamp": all_timestamps,
        "status": all_statuses,
        "runtime_ms": all_runtimes,
        "language": all_languages,
    })

    return df[SUBMISSION_COLUMNS]


def generate_synthetic_dataset_v3(
    questions_df: pd.DataFrame | None = None,
    n_users: int = 2000,
    random_seed: int = config.RANDOM_SEED,
    beta: float = 1.5,
    p_retry: float = 0.35,
    mean_submissions: float = 100.0,
    min_submissions: int = 20,
    ref_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience entry point returning (users_df, questions_df, submissions_df, latent_df)."""
    if questions_df is None:
        questions_df = load_canonical_questions()
    users_df, latent_df = generate_synthetic_users_v3(
        n=n_users, random_seed=random_seed, return_latent_info=True
    )
    submissions_df = generate_synthetic_submissions_v3(
        users_df=users_df,
        questions_df=questions_df,
        random_seed=random_seed,
        latent_users_info=latent_df,
        beta=beta,
        p_retry=p_retry,
        mean_submissions=mean_submissions,
        min_submissions=min_submissions,
        ref_date=ref_date,
    )
    return users_df, questions_df, submissions_df, latent_df


if __name__ == "__main__":
    users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
    fe = FeatureEngineer(users_df, questions_df, submissions_df)
    feature_matrix = fe.build_user_feature_matrix()
    print(feature_matrix.head())
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
