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
    questions = (
        submissions_df[["question_id", "title", "difficulty", "topic_tags", "acceptance_rate"]]
        .drop_duplicates(subset=["question_id"])
        .copy()
    )
    if "description" not in questions.columns:
        questions["description"] = (
            questions["title"].astype(str)
            + " "
            + questions["difficulty"].astype(str)
            + " problem statement"
        )
    questions = questions.reset_index(drop=True)
    questions["question_key"] = questions["question_id"]
    questions["question_id"] = np.arange(1, len(questions) + 1)
    question_id_map = dict(zip(questions["question_key"], questions["question_id"]))
    return questions[QUESTION_COLUMNS], question_id_map


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

    # Augment the question bank if the export contains only attempted items.
    if len(questions_df) < 50:
        synthetic_questions = generate_synthetic_questions(n=50)
        start_id = int(questions_df["question_id"].max()) + 1 if not questions_df.empty else 1
        synthetic_questions = synthetic_questions.assign(
            question_id=np.arange(start_id, start_id + len(synthetic_questions))
        )
        questions_df = pd.concat([questions_df, synthetic_questions], ignore_index=True)

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

    if len(questions_df) < 50:
        synthetic_questions = generate_synthetic_questions(n=50)
        start_id = int(questions_df["question_id"].max()) + 1 if not questions_df.empty else 1
        synthetic_questions = synthetic_questions.assign(
            question_id=np.arange(start_id, start_id + len(synthetic_questions))
        )
        questions_df = pd.concat([questions_df, synthetic_questions], ignore_index=True)

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
        for col in ["accuracy_easy", "accuracy_medium", "accuracy_hard"]:
            if col not in per_difficulty.columns:
                per_difficulty[col] = 0.0

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
        for col in ["share_easy", "share_medium", "share_hard"]:
            if col not in shares.columns:
                shares[col] = 0.0
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
            # Skill & Risk Composite Score
            risk_score = (
                0.40 * (1.0 - success_rate)
                + 0.30 * (1.0 - recency_success)
                + 0.20 * easy_fail_rate
                + 0.10 * (1.0 - min(1.0, unique_attempted / 10.0))
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


if __name__ == "__main__":
    users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
    fe = FeatureEngineer(users_df, questions_df, submissions_df)
    feature_matrix = fe.build_user_feature_matrix()
    print(feature_matrix.head())
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
