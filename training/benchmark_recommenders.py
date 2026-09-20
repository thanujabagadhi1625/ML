"""
training/benchmark_recommenders.py
----------------------------------
Honest, multi-seed evaluation harness for recommendation models:
  a) random among unsolved questions
  b) popularity: most-solved questions among TRAIN users, excluding user's solved questions
  c) content-only (tag similarity)
  d) ALS-only (fold-in scores)
  e) hybrid (HybridRecommender)

Metrics reported at K=5 and K=10:
  - Precision@K, Recall@K, HitRate@K, NDCG@K
  - Per-method mean and std across seeds
  - 95% bootstrap confidence interval over user evaluations
  - Paired differences (hybrid - popularity, hybrid - random) with 95% CI
  - Evaluated test users count, ground-truth items per user
  - ALS training time, average interactions per user/item, matrix density
  - Head vs tail item breakdown (top 20% items by popularity vs remaining 80%)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add files and project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

import config
from data_processing import (
    generate_synthetic_submissions,
    generate_synthetic_users,
    load_canonical_questions,
    temporal_split_user_submissions,
    user_train_val_test_split,
)
from recommender import (
    ALSMatrixFactorization,
    ContentBasedRecommender,
    HybridRecommender,
    build_interaction_matrix,
)


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes a two-sided bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return (round(lower, 5), round(upper, 5))


def get_popularity_rankings(train_subs: pd.DataFrame, all_qids: list[int]) -> list[int]:
    """Returns question IDs sorted descending by number of train users who solved them."""
    train_solved = train_subs[train_subs["status"] == "Accepted"]
    if train_solved.empty:
        return sorted(all_qids)
    solve_counts = train_solved.groupby("question_id")["user_id"].nunique().to_dict()
    # Sort descending by solve count, break ties by question_id ascending
    return sorted(all_qids, key=lambda q: (-solve_counts.get(q, 0), q))


def run_benchmark(
    seeds: list[int] = list(range(1, 11)),
    k_values: list[int] = [5, 10],
    generator_version: str = "v1",
    catalogue_path: str | Path | None = None,
    n_users: int = config.NUM_USERS,
    n_submissions: int = config.NUM_SUBMISSIONS,
    output_path: str | Path | None = None,
    als_params: dict | None = None,
    hybrid_weights: tuple[float, float, float] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Executes the full multi-seed recommendation benchmark across candidate methods.
    """
    if als_params is None:
        als_params = {
            "n_factors": config.MF_LATENT_DIM,
            "reg_lambda": config.MF_REG_LAMBDA,
            "alpha": config.MF_CONFIDENCE_ALPHA,
            "n_epochs": config.MF_EPOCHS,
        }

    # Load questions catalogue
    if catalogue_path is not None:
        c_path = Path(catalogue_path)
        if not c_path.exists():
            raise FileNotFoundError(f"Catalogue file not found at {c_path}")
        records = json.loads(c_path.read_text(encoding="utf-8"))
        questions_df = pd.DataFrame(records)[config.QUESTION_COLUMNS].reset_index(drop=True)
    else:
        questions_df = load_canonical_questions().reset_index(drop=True)

    all_qids = questions_df["question_id"].tolist()
    content_model = ContentBasedRecommender(questions_df)
    methods = ["random", "popularity", "content_only", "als_only", "hybrid"]

    # Storage for per-seed and per-user metrics
    seed_metrics: dict[str, dict[str, list[float]]] = {
        m: {f"{metric}@{k}": [] for k in k_values for metric in ["precision", "recall", "hit_rate", "ndcg"]}
        for m in methods
    }
    user_evaluations: dict[str, dict[str, list[float]]] = {
        m: {f"{metric}@{k}": [] for k in k_values for metric in ["precision", "recall", "hit_rate", "ndcg"]}
        for m in methods
    }

    # Head vs tail tracking for hybrid and popularity
    head_tail_shares: dict[str, dict[int, list[float]]] = {
        m: {k: [] for k in k_values} for m in methods
    }

    evaluated_users_per_seed = []
    gt_items_per_user_per_seed = []
    als_training_times = []
    interactions_per_user_list = []
    interactions_per_item_list = []
    matrix_densities = []

    for seed in seeds:
        if verbose:
            print(f"\n--- Running Seed {seed}/{len(seeds)} (Generator: {generator_version}) ---")

        # 1. Generate data
        if generator_version == "v1":
            users_df, latent_info = generate_synthetic_users(
                n=n_users, random_seed=seed, return_latent_info=True
            )
            subs_df = generate_synthetic_submissions(
                users_df,
                questions_df,
                n_submissions=n_submissions,
                random_seed=seed,
                latent_users_info=latent_info,
            )
        elif generator_version == "v3":
            from data_processing import generate_synthetic_dataset_v3
            users_df, questions_df_gen, subs_df, _ = generate_synthetic_dataset_v3(
                questions_df=questions_df,
                n_users=n_users,
                random_seed=seed,
            )
            # Ensure questions_df aligns
            questions_df = questions_df_gen
            all_qids = questions_df["question_id"].tolist()
            content_model = ContentBasedRecommender(questions_df)
        else:
            raise ValueError(f"Unsupported generator version: {generator_version}")

        # 2. Split users 70/15/15
        train_u, val_u, test_u, train_s, val_s, test_s = user_train_val_test_split(
            users_df, subs_df, random_seed=seed
        )

        # 3. Train ALS on train users only
        t_start = time.perf_counter()
        train_interactions = build_interaction_matrix(train_s)
        als = ALSMatrixFactorization(
            n_factors=als_params["n_factors"],
            reg_lambda=als_params["reg_lambda"],
            alpha=als_params["alpha"],
            n_epochs=als_params["n_epochs"],
            random_state=seed,
        ).fit(train_interactions)
        t_als = time.perf_counter() - t_start
        als_training_times.append(t_als)

        # Dataset statistics
        n_train_users = train_u["user_id"].nunique()
        n_catalogue_items = len(questions_df)
        n_interactions = len(train_interactions)
        avg_inter_user = n_interactions / max(n_train_users, 1)
        avg_inter_item = n_interactions / max(n_catalogue_items, 1)
        density = n_interactions / max(n_train_users * n_catalogue_items, 1)
        interactions_per_user_list.append(avg_inter_user)
        interactions_per_item_list.append(avg_inter_item)
        matrix_densities.append(density)

        # Precompute popularity ordering from train
        pop_ordered = get_popularity_rankings(train_s, all_qids)

        # Head items definition: top 20% most-solved items in train
        n_head = max(1, int(round(0.20 * len(all_qids))))
        head_qids = set(pop_ordered[:n_head])

        # 4. Temporal split test users (80% history / 20% future)
        test_h, test_f = temporal_split_user_submissions(
            test_s, history_ratio=config.TEMPORAL_HISTORY_RATIO
        )

        # Build ground truth
        future_solved_by_user = (
            test_f[test_f["status"] == "Accepted"]
            .groupby("user_id")["question_id"]
            .apply(set)
            .to_dict()
        )
        history_solved_by_user = (
            test_h[test_h["status"] == "Accepted"]
            .groupby("user_id")["question_id"]
            .apply(set)
            .to_dict()
        )

        # Initialize HybridRecommender
        recommender = HybridRecommender(questions_df, als_model=als)
        if hybrid_weights is not None:
            # Dynamically set weights if custom tuned
            w_cf, w_content, w_weak = hybrid_weights
            recommender.cf_weight = w_cf
            recommender.content_weight = w_content
            recommender.weakness_weight = w_weak

        # Per-seed accumulator for this seed's user means
        seed_user_metrics: dict[str, dict[str, list[float]]] = {
            m: {f"{metric}@{k}": [] for k in k_values for metric in ["precision", "recall", "hit_rate", "ndcg"]}
            for m in methods
        }

        test_uids = sorted(test_u["user_id"].unique())
        evaluated_users_count = 0
        gt_item_counts = []

        # Recommender evaluation per user
        for uid in test_uids:
            past_solved = history_solved_by_user.get(uid, set())
            future_solved = future_solved_by_user.get(uid, set())
            gt_solved = future_solved - past_solved
            if not gt_solved:
                continue

            evaluated_users_count += 1
            gt_item_counts.append(len(gt_solved))
            u_hist = test_h[test_h["user_id"] == uid]

            # Candidate pool excluding past_solved
            unsolved_candidates = [q for q in all_qids if q not in past_solved]

            # a) Random
            user_rand_rng = np.random.default_rng(seed * 100_000 + int(uid))
            random_pool = list(unsolved_candidates)
            user_rand_rng.shuffle(random_pool)

            # b) Popularity
            pop_recs = [q for q in pop_ordered if q not in past_solved]

            # c) Content-Only
            solved_ids = list(past_solved)
            failed_ids = u_hist[u_hist["status"] != "Accepted"]["question_id"].unique().tolist()
            content_scores = content_model.similar_to_user_profile(solved_ids, failed_ids)
            content_scores_by_qid = dict(zip(questions_df["question_id"], content_scores))
            content_recs = sorted(unsolved_candidates, key=lambda q: (-content_scores_by_qid.get(q, 0.0), q))

            # d) ALS-Only
            if not u_hist.empty:
                interactions = build_interaction_matrix(u_hist)
                raw_cf, cf_ok = als.fold_in_user(
                    interactions["question_id"].tolist(),
                    interactions["weight"].tolist(),
                )
                if cf_ok:
                    cf_by_qid = dict(zip(als.item_ids_ordered(), raw_cf))
                    als_recs = sorted(unsolved_candidates, key=lambda q: (-cf_by_qid.get(q, 0.0), q))
                else:
                    als_recs = list(pop_recs)
            else:
                als_recs = list(pop_recs)

            # e) Hybrid (current HybridRecommender)
            hybrid_recs_df_max = recommender.recommend(
                user_id=uid, top_n=max(k_values), user_submissions_df=u_hist
            )
            hybrid_recs = hybrid_recs_df_max["question_id"].tolist()

            method_recs = {
                "random": random_pool,
                "popularity": pop_recs,
                "content_only": content_recs,
                "als_only": als_recs,
                "hybrid": hybrid_recs,
            }

            for m in methods:
                for k in k_values:
                    top_k = method_recs[m][:k]
                    # Track head share
                    n_head_in_k = sum(1 for q in top_k if q in head_qids)
                    head_tail_shares[m][k].append(n_head_in_k / float(k))

                    matched = [1 if q in gt_solved else 0 for q in top_k]
                    n_matched = sum(matched)
                    prec = n_matched / float(k)
                    rec = n_matched / float(len(gt_solved))
                    hit = 1.0 if n_matched > 0 else 0.0
                    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(matched))
                    ideal_hits = min(k, len(gt_solved))
                    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
                    ndcg = dcg / idcg if idcg > 0 else 0.0

                    m_key = f"precision@{k}"
                    seed_user_metrics[m][m_key].append(prec)
                    user_evaluations[m][m_key].append(prec)

                    m_key = f"recall@{k}"
                    seed_user_metrics[m][m_key].append(rec)
                    user_evaluations[m][m_key].append(rec)

                    m_key = f"hit_rate@{k}"
                    seed_user_metrics[m][m_key].append(hit)
                    user_evaluations[m][m_key].append(hit)

                    m_key = f"ndcg@{k}"
                    seed_user_metrics[m][m_key].append(ndcg)
                    user_evaluations[m][m_key].append(ndcg)

        evaluated_users_per_seed.append(evaluated_users_count)
        gt_items_per_user_per_seed.append(float(np.mean(gt_item_counts)) if gt_item_counts else 0.0)

        # Average user metrics for this seed
        for m in methods:
            for metric_key in seed_user_metrics[m]:
                vals = seed_user_metrics[m][metric_key]
                seed_metrics[m][metric_key].append(float(np.mean(vals)) if vals else 0.0)

        if verbose:
            print(f"  Evaluated Users: {evaluated_users_count} | Avg GT items/user: {gt_items_per_user_per_seed[-1]:.2f}")
            print(f"  Hybrid P@5: {seed_metrics['hybrid']['precision@5'][-1]:.4f} | Popularity P@5: {seed_metrics['popularity']['precision@5'][-1]:.4f} | Random P@5: {seed_metrics['random']['precision@5'][-1]:.4f}")

    # Compute aggregate summary
    summary_results: dict[str, Any] = {
        "metadata": {
            "generator_version": generator_version,
            "catalogue_path": str(catalogue_path) if catalogue_path else "default_canonical",
            "n_users": n_users,
            "n_questions": len(questions_df),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "k_values": k_values,
            "evaluated_users_total": len(user_evaluations["hybrid"]["precision@5"]),
            "evaluated_users_mean_per_seed": float(np.mean(evaluated_users_per_seed)),
            "gt_items_per_user_mean": float(np.mean(gt_items_per_user_per_seed)),
            "als_training_time_mean_seconds": float(np.mean(als_training_times)),
            "interactions_per_user_mean": float(np.mean(interactions_per_user_list)),
            "interactions_per_item_mean": float(np.mean(interactions_per_item_list)),
            "matrix_density_mean": float(np.mean(matrix_densities)),
        },
        "methods": {},
        "paired_differences": {},
        "head_vs_tail": {},
    }

    for m in methods:
        summary_results["methods"][m] = {}
        for k in k_values:
            for metric in ["precision", "recall", "hit_rate", "ndcg"]:
                key = f"{metric}@{k}"
                seed_vals = seed_metrics[m][key]
                user_vals = user_evaluations[m][key]
                ci_low, ci_high = bootstrap_ci(user_vals, n_boot=2000, seed=42)
                summary_results["methods"][m][key] = {
                    "mean": round(float(np.mean(seed_vals)), 5),
                    "std": round(float(np.std(seed_vals, ddof=1)), 5) if len(seed_vals) > 1 else 0.0,
                    "ci_95": [ci_low, ci_high],
                }

    # Paired differences: hybrid - popularity, hybrid - random
    for comp_method in ["popularity", "random"]:
        pair_key = f"hybrid_minus_{comp_method}"
        summary_results["paired_differences"][pair_key] = {}
        for k in k_values:
            for metric in ["precision", "recall", "hit_rate", "ndcg"]:
                key = f"{metric}@{k}"
                # Per seed paired difference
                seed_diffs = [
                    h - p
                    for h, p in zip(seed_metrics["hybrid"][key], seed_metrics[comp_method][key])
                ]
                # Per user paired difference
                user_diffs = [
                    h - p
                    for h, p in zip(user_evaluations["hybrid"][key], user_evaluations[comp_method][key])
                ]
                ci_low, ci_high = bootstrap_ci(user_diffs, n_boot=2000, seed=42)
                summary_results["paired_differences"][pair_key][key] = {
                    "mean": round(float(np.mean(seed_diffs)), 5),
                    "std": round(float(np.std(seed_diffs, ddof=1)), 5) if len(seed_diffs) > 1 else 0.0,
                    "ci_95": [ci_low, ci_high],
                }

    # Head vs tail recommendation share
    for m in methods:
        summary_results["head_vs_tail"][m] = {}
        for k in k_values:
            shares = head_tail_shares[m][k]
            summary_results["head_vs_tail"][m][f"head_share@{k}"] = {
                "mean": round(float(np.mean(shares)), 4) if shares else 0.0,
                "std": round(float(np.std(shares)), 4) if shares else 0.0,
            }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
        if verbose:
            print(f"\nSaved benchmark results to {out_p}")

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed Recommendation Benchmark")
    parser.add_argument("--generator-version", choices=["v1", "v3"], default="v1")
    parser.add_argument("--catalogue-path", type=str, default=None)
    parser.add_argument("--n-users", type=int, default=config.NUM_USERS)
    parser.add_argument("--n-submissions", type=int, default=config.NUM_SUBMISSIONS)
    parser.add_argument("--output", type=str, default="results/bench_v1.json")
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()

    seeds = list(range(1, args.n_seeds + 1))
    run_benchmark(
        seeds=seeds,
        k_values=[5, 10],
        generator_version=args.generator_version,
        catalogue_path=args.catalogue_path,
        n_users=args.n_users,
        n_submissions=args.n_submissions,
        output_path=args.output,
        verbose=True,
    )
