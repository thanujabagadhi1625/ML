"""
training/train_als_real.py
--------------------------
Optional pipeline for training ALS on a real interaction dataset (e.g., IBM Project CodeNet,
or real LeetCode multi-user interaction dump) when available.

Usage:
    python training/train_als_real.py --data-path data/real_interactions.csv
    python training/train_als_real.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add files/ and project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

import numpy as np
import pandas as pd

import config
from data_processing import (
    load_canonical_questions,
    temporal_split_user_submissions,
    user_train_val_test_split,
)
from recommender import (
    ALSMatrixFactorization,
    HybridRecommender,
    build_interaction_matrix,
)


def load_real_interactions(
    data_path: Path,
    canonical_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingests and maps real interactions from a CSV/JSON file to the canonical question catalogue.
    Expected input columns (flexible):
      - user_id / user / author
      - question_id / problem_id / slug / title
      - status / result (or weight / points)
      - timestamp / submission_time (optional, defaults to synthetic ordering if missing)
    """
    print(f"Reading real interaction data from '{data_path}'...")
    if data_path.suffix.lower() == ".json":
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("submissions") or raw.get("data") or raw
        df = pd.DataFrame(raw)
    else:
        df = pd.read_csv(data_path)

    print(f"  Raw rows read: {len(df)}")

    # Standardize column names
    col_map = {
        "author": "user_id", "user": "user_id", "username": "user_id",
        "problem_id": "question_id", "problem_slug": "slug", "title_slug": "slug",
        "result": "status", "submission_status": "status", "verdict": "status",
        "submission_time": "timestamp", "time": "timestamp", "date": "timestamp",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "user_id" not in df.columns or "status" not in df.columns:
        raise ValueError("Real interaction data must contain user and status/outcome columns.")

    # Build canonical mapping tables
    slug_map = dict(zip(canonical_df["slug"].astype(str).str.lower(), canonical_df["question_id"]))
    title_map = dict(zip(canonical_df["title"].astype(str).str.lower(), canonical_df["question_id"]))
    leetcode_id_map = {}
    if "leetcode_id" in canonical_df.columns:
        for lid, qid in zip(canonical_df["leetcode_id"], canonical_df["question_id"]):
            if pd.notna(lid):
                try:
                    leetcode_id_map[int(float(lid))] = qid
                except (ValueError, TypeError):
                    pass

    # Align question IDs to canonical IDs
    mapped_qids = []
    dropped_unmapped = 0
    for _, row in df.iterrows():
        raw_q = str(row.get("question_id") or row.get("slug") or row.get("title") or "").strip().lower()
        qid = None
        if raw_q in slug_map:
            qid = slug_map[raw_q]
        elif raw_q in title_map:
            qid = title_map[raw_q]
        elif raw_q.isdigit() and int(raw_q) in leetcode_id_map:
            qid = leetcode_id_map[int(raw_q)]
        elif raw_q.isdigit() and int(raw_q) in set(canonical_df["question_id"]):
            qid = int(raw_q)

        if qid is not None:
            mapped_qids.append(qid)
        else:
            mapped_qids.append(None)
            dropped_unmapped += 1

    df["question_id"] = mapped_qids
    df = df.dropna(subset=["question_id"]).copy()
    df["question_id"] = df["question_id"].astype(int)

    print(f"  Mapped interactions to catalogue: {len(df)} rows ({dropped_unmapped} dropped unmapped)")
    print(f"  Unique users: {df['user_id'].nunique()}, Unique questions: {df['question_id'].nunique()}")

    # Ensure timestamp is datetime
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="min")
    elif not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(pd.Timestamp("2026-01-01"))

    # Map status to Accepted / Failed
    status_str = df["status"].astype(str).str.lower()
    df["status"] = np.where(status_str.str.contains("accept|correct|10|pass|ac"), "Accepted", "Wrong Answer")

    # Build users table
    unique_users = df["user_id"].unique()
    users_df = pd.DataFrame({
        "user_id": unique_users,
        "account_age_days": 365,
        "latent_skill": 0.0,
        "contest_rating": 1500.0,
    })

    return users_df, df


def train_and_eval_real(
    data_path: Path,
    output_path: Path | None = None,
    n_factors: int = 16,
    alpha: float = 5.0,
    reg_lambda: float = 50.0,
    epochs: int = 15,
) -> dict:
    canonical_df = load_canonical_questions()
    all_qids = canonical_df["question_id"].tolist()

    users_df, subs_df = load_real_interactions(data_path, canonical_df)

    # User-level split
    train_u, val_u, test_u, train_s, val_s, test_s = user_train_val_test_split(
        users_df, subs_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )
    test_h, test_f = temporal_split_user_submissions(test_s, history_ratio=0.80)

    # Fit ALS with all canonical item IDs preserved
    train_inter = build_interaction_matrix(train_s)
    print(f"\nFitting ALS on {len(train_inter)} real user-item interaction pairs...")
    als = ALSMatrixFactorization(
        n_factors=n_factors,
        reg_lambda=reg_lambda,
        alpha=alpha,
        n_epochs=epochs,
        random_state=42,
    )
    t0 = time.time()
    als.fit(train_inter, all_item_ids=all_qids)
    print(f"  Fitted in {time.time() - t0:.2f}s.")

    # Evaluate on held-out test users
    rec = HybridRecommender(canonical_df, als_model=als)
    test_uids = test_u["user_id"].tolist()
    eval_res = rec.evaluate_recommendations(rec, test_uids, test_h, test_f, top_n=5)
    print("\nReal Dataset Held-Out Evaluation (Test Users):")
    print(f"  Precision@5: {eval_res.get('precision@5', 0):.4f}")
    print(f"  Recall@5:    {eval_res.get('recall@5', 0):.4f}")
    print(f"  Hit Rate@5:  {eval_res.get('hit_rate@5', 0):.4f}")
    print(f"  NDCG@5:      {eval_res.get('ndcg@5', 0):.4f}")
    print(f"  Evaluated:   {eval_res.get('evaluated_users', 0)} users")

    if output_path:
        als.save(output_path)
        print(f"Saved real-trained ALS model to {output_path}")

    return eval_res


def main():
    parser = argparse.ArgumentParser(description="Train ALS Collaborative Filtering on Real Interaction Data")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT_DIR / "data" / "real_interactions.csv",
        help="Path to real interaction CSV or JSON (e.g. CodeNet extract).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=ROOT_DIR / "models" / "als_model_real.npz",
        help="Target path to save the real-trained ALS model.",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Real interaction dataset not found at '{args.data_path}'.")
        print("To run real-dataset training, provide a CSV with user_id, question_id/slug, and status/verdict.")
        print("Example usage: python training/train_als_real.py --data-path path/to/interactions.csv")
        return

    train_and_eval_real(args.data_path, args.output_model)


if __name__ == "__main__":
    main()
