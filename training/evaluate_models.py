"""
training/evaluate_models.py
---------------------------
Loads serialized model artifacts from models/ and evaluates performance on held-out test users:
  1. XGBoost Rating Predictor: RMSE, MAE, R^2
  2. Weak Topic Detection: Precision, Recall, F1
  3. Recommender Honest Baseline Comparison:
     - Methods: Hybrid Recommender (Production), Popularity, Content-Only, Random
     - Overall metrics: Precision@5, Recall@5, Hit Rate@5, NDCG@5
     - Long-tail metrics: Hit Rate@5 (tail), NDCG@5 (tail), Precision@5 (tail), Recall@5 (tail)
       (evaluated strictly on held-out items OUTSIDE the top 20% most popular problems)
     - Popularity-debiased comparison: IPS-NDCG@5 (Inverse Propensity Scored NDCG@5)
     - Catalog discovery: Tail Share@5 (% of recommended items outside top 20% head)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Add project root and files/ to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

import config
from data_processing import (
    FeatureEngineer,
    compute_user_topic_profile,
    generate_synthetic_dataset_v3,
    generate_synthetic_submissions,
    generate_synthetic_users,
    load_canonical_questions,
    temporal_split_user_submissions,
    user_train_val_test_split,
)
from predictor import ContestRatingPredictor
from recommender import (
    ALSMatrixFactorization,
    ContentBasedRecommender,
    HybridRecommender,
)
from training.train_models import evaluate_weak_topic_detection


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes a two-sided bootstrap confidence interval for the mean with n_boot resamples."""
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


def evaluate_recommenders_comprehensive(
    questions_df: pd.DataFrame,
    train_subs: pd.DataFrame,
    test_users: pd.DataFrame,
    test_hist_subs: pd.DataFrame,
    test_fut_subs: pd.DataFrame,
    als_model: ALSMatrixFactorization,
    top_k: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates Hybrid Recommender against Popularity, Content-Only, and Random baselines
    across overall, long-tail (outside top 20% head), and popularity-debiased (IPS) metrics.
    """
    all_qids = questions_df["question_id"].tolist()
    n_catalogue = len(all_qids)

    # 1. Determine item popularity from train accepted submissions
    train_solved = train_subs[train_subs["status"] == "Accepted"]
    train_solve_counts = train_solved.groupby("question_id")["user_id"].nunique().to_dict()
    pop_ordered = sorted(all_qids, key=lambda q: (-train_solve_counts.get(q, 0), q))

    # Top 20% head items vs remaining 80% long-tail items
    n_head = max(1, int(round(0.20 * n_catalogue)))
    head_qids: Set[int] = set(pop_ordered[:n_head])
    tail_qids: Set[int] = set(all_qids) - head_qids

    # 2. Propensity model for debiasing (IPS-NDCG)
    # Propensity p_q proportional to sqrt(popularity); weight w_q = 1 / p_q
    max_solves = max(train_solve_counts.values()) if train_solve_counts else 1
    propensities = {
        q: (max(train_solve_counts.get(q, 0), 1) / max_solves) ** 0.5
        for q in all_qids
    }
    ips_weights = {
        q: min(1.0 / max(propensities[q], 0.01), 20.0)
        for q in all_qids
    }

    # 3. Initialize models
    hybrid_recommender = HybridRecommender(questions_df, als_model=als_model)
    hybrid_recommender.cf_weight = getattr(config, "HYBRID_CF_WEIGHT", 0.50)
    hybrid_recommender.content_weight = getattr(config, "HYBRID_CONTENT_WEIGHT", 0.25)
    hybrid_recommender.weakness_weight = getattr(config, "HYBRID_WEAKNESS_WEIGHT", 0.25)

    discovery_recommender = HybridRecommender(questions_df, als_model=als_model)
    discovery_recommender.cf_weight = getattr(config, "DISCOVERY_CF_WEIGHT", 0.00)
    discovery_recommender.content_weight = getattr(config, "DISCOVERY_CONTENT_WEIGHT", 0.80)
    discovery_recommender.weakness_weight = getattr(config, "DISCOVERY_WEAKNESS_WEIGHT", 0.20)

    content_recommender = ContentBasedRecommender(questions_df)

    # 4. Extract ground truth per test user
    future_solved_by_user = (
        test_fut_subs[test_fut_subs["status"] == "Accepted"]
        .groupby("user_id")["question_id"]
        .apply(set)
        .to_dict()
    )
    history_solved_by_user = (
        test_hist_subs[test_hist_subs["status"] == "Accepted"]
        .groupby("user_id")["question_id"]
        .apply(set)
        .to_dict()
    )

    test_uids = sorted(test_users["user_id"].unique())

    methods = ["hybrid", "discovery", "popularity", "content_only", "random"]
    metric_accumulators: Dict[str, Dict[str, List[float]]] = {
        m: {
            "precision": [],
            "recall": [],
            "hit_rate": [],
            "ndcg": [],
            "tail_hit_rate": [],
            "tail_precision": [],
            "tail_recall": [],
            "tail_ndcg": [],
            "ips_ndcg": [],
            "tail_share": [],
        }
        for m in methods
    }

    evaluated_users = 0
    users_with_tail_gt = 0

    for uid in test_uids:
        past_solved = history_solved_by_user.get(uid, set())
        future_solved = future_solved_by_user.get(uid, set())
        gt_all = future_solved - past_solved
        if not gt_all:
            continue

        gt_tail = gt_all - head_qids
        has_tail_gt = len(gt_tail) > 0

        evaluated_users += 1
        if has_tail_gt:
            users_with_tail_gt += 1

        u_hist = test_hist_subs[test_hist_subs["user_id"] == uid]
        unsolved_pool = [q for q in all_qids if q not in past_solved]

        # Topic profile for hybrid
        u_topic_profile = compute_user_topic_profile(u_hist, questions_df, user_id=uid)

        # Generate top_k recommendations for each method
        # (a) Hybrid Recommender
        rec_hybrid_df = hybrid_recommender.recommend(
            user_id=uid,
            top_n=top_k,
            topic_profile=u_topic_profile,
            user_submissions_df=u_hist,
            enforce_diversity=True,
        )
        hybrid_recs = rec_hybrid_df["question_id"].tolist()

        # (b) Popularity Baseline
        pop_recs = [q for q in pop_ordered if q not in past_solved][:top_k]

        # (c) Content-Only Baseline
        solved_ids = list(past_solved)
        failed_ids = u_hist[u_hist["status"] != "Accepted"]["question_id"].unique().tolist()
        content_scores = content_recommender.similar_to_user_profile(solved_ids, failed_ids)
        score_by_qid = dict(zip(questions_df["question_id"], content_scores))
        content_recs = sorted(unsolved_pool, key=lambda q: (-score_by_qid.get(q, 0.0), q))[:top_k]

        # (d) Random Baseline
        user_rng = np.random.default_rng(random_seed * 10_000 + int(uid))
        rand_recs = list(user_rng.choice(unsolved_pool, size=min(top_k, len(unsolved_pool)), replace=False))

        rec_disc_df = discovery_recommender.recommend(
            user_id=uid,
            top_n=top_k,
            topic_profile=u_topic_profile,
            user_submissions_df=u_hist,
            enforce_diversity=True,
        )
        disc_recs = rec_disc_df["question_id"].tolist()

        method_recs = {
            "hybrid": hybrid_recs,
            "discovery": disc_recs,
            "popularity": pop_recs,
            "content_only": content_recs,
            "random": rand_recs,
        }

        # Calculate metrics for each method
        for m, recs in method_recs.items():
            top_recs = recs[:top_k]

            # 1. Overall metrics
            hits = [1 if q in gt_all else 0 for q in top_recs]
            n_hits = sum(hits)
            prec = n_hits / float(top_k)
            rec = n_hits / float(len(gt_all))
            hit_r = 1.0 if n_hits > 0 else 0.0

            dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
            idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(top_k, len(gt_all))))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            # 2. Long-tail metrics (on held-out items OUTSIDE top 20% head)
            tail_share = sum(1 for q in top_recs if q in tail_qids) / float(top_k)
            metric_accumulators[m]["tail_share"].append(tail_share)

            if has_tail_gt:
                tail_hits = [1 if q in gt_tail else 0 for q in top_recs]
                n_tail_hits = sum(tail_hits)
                tail_prec = n_tail_hits / float(top_k)
                tail_rec = n_tail_hits / float(len(gt_tail))
                tail_hit_r = 1.0 if n_tail_hits > 0 else 0.0

                dcg_tail = sum(h / np.log2(idx + 2) for idx, h in enumerate(tail_hits))
                idcg_tail = sum(1.0 / np.log2(idx + 2) for idx in range(min(top_k, len(gt_tail))))
                tail_ndcg = dcg_tail / idcg_tail if idcg_tail > 0 else 0.0

                metric_accumulators[m]["tail_hit_rate"].append(tail_hit_r)
                metric_accumulators[m]["tail_precision"].append(tail_prec)
                metric_accumulators[m]["tail_recall"].append(tail_rec)
                metric_accumulators[m]["tail_ndcg"].append(tail_ndcg)

            # 3. Popularity-debiased metric (IPS-NDCG@5)
            dcg_ips = sum(
                (ips_weights.get(q, 1.0) / np.log2(idx + 2)) if q in gt_all else 0.0
                for idx, q in enumerate(top_recs)
            )
            # Ideal DCG: top items in gt_all sorted descending by IPS weight
            sorted_gt_ips = sorted([ips_weights.get(q, 1.0) for q in gt_all], reverse=True)[:top_k]
            idcg_ips = sum(w / np.log2(idx + 2) for idx, w in enumerate(sorted_gt_ips))
            ips_ndcg = dcg_ips / idcg_ips if idcg_ips > 0 else 0.0

            metric_accumulators[m]["precision"].append(prec)
            metric_accumulators[m]["recall"].append(rec)
            metric_accumulators[m]["hit_rate"].append(hit_r)
            metric_accumulators[m]["ndcg"].append(ndcg)
            metric_accumulators[m]["ips_ndcg"].append(ips_ndcg)

    results: Dict[str, Any] = {
        "metadata": {
            "evaluated_users": evaluated_users,
            "users_with_tail_gt": users_with_tail_gt,
            "catalogue_size": n_catalogue,
            "head_cutoff_top20pct": n_head,
            "top_k": top_k,
        },
        "methods": {},
    }

    for m in methods:
        acc = metric_accumulators[m]
        hit_ci = bootstrap_ci(acc["hit_rate"], n_boot=1000, seed=42)
        ndcg_ci = bootstrap_ci(acc["ndcg"], n_boot=1000, seed=42)
        results["methods"][m] = {
            "precision@5": round(float(np.mean(acc["precision"])), 5) if acc["precision"] else 0.0,
            "recall@5": round(float(np.mean(acc["recall"])), 5) if acc["recall"] else 0.0,
            "hit_rate@5": round(float(np.mean(acc["hit_rate"])), 5) if acc["hit_rate"] else 0.0,
            "hit_rate@5_ci95": [hit_ci[0], hit_ci[1]],
            "ndcg@5": round(float(np.mean(acc["ndcg"])), 5) if acc["ndcg"] else 0.0,
            "ndcg@5_ci95": [ndcg_ci[0], ndcg_ci[1]],
            "tail_hit_rate@5": round(float(np.mean(acc["tail_hit_rate"])), 5) if acc["tail_hit_rate"] else 0.0,
            "tail_precision@5": round(float(np.mean(acc["tail_precision"])), 5) if acc["tail_precision"] else 0.0,
            "tail_recall@5": round(float(np.mean(acc["tail_recall"])), 5) if acc["tail_recall"] else 0.0,
            "tail_ndcg@5": round(float(np.mean(acc["tail_ndcg"])), 5) if acc["tail_ndcg"] else 0.0,
            "ips_ndcg@5": round(float(np.mean(acc["ips_ndcg"])), 5) if acc["ips_ndcg"] else 0.0,
            "tail_share@5": round(float(np.mean(acc["tail_share"])), 5) if acc["tail_share"] else 0.0,
        }

    # Compute paired difference bootstrap CIs for hybrid vs popularity
    diff_hit = np.array(metric_accumulators["hybrid"]["hit_rate"]) - np.array(metric_accumulators["popularity"]["hit_rate"])
    diff_ndcg = np.array(metric_accumulators["hybrid"]["ndcg"]) - np.array(metric_accumulators["popularity"]["ndcg"])
    results["paired_differences"] = {
        "hybrid_minus_popularity": {
            "hit_rate@5_diff": round(float(np.mean(diff_hit)), 5),
            "hit_rate@5_diff_ci95": list(bootstrap_ci(diff_hit, n_boot=1000, seed=42)),
            "ndcg@5_diff": round(float(np.mean(diff_ndcg)), 5),
            "ndcg@5_diff_ci95": list(bootstrap_ci(diff_ndcg, n_boot=1000, seed=42)),
        }
    }

    return results


def run_evaluation():
    print("=" * 80)
    print("LEETCODE MENTOR — COMPREHENSIVE ARTIFACT & BASELINE EVALUATION")
    print("=" * 80)

    if not config.MODEL_METADATA_PATH.exists():
        print(f"Error: Model metadata not found at '{config.MODEL_METADATA_PATH}'.")
        print("Please run offline training first: python training/train_models.py")
        return

    metadata = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    print(f"\nModel Version:        {metadata.get('model_version')}")
    print(f"Dataset Version:      {metadata.get('dataset_version', 'N/A')}")
    print(f"Trained At:           {metadata.get('training_timestamp')}")
    print(f"Active Profile:       {config.CURRENT_PROFILE}")
    print(f"Catalogue Size:       {metadata.get('n_questions', config.NUM_QUESTIONS)} problems")
    print(f"Benchmark Population: {metadata.get('n_synthetic_users', config.NUM_USERS)} synthetic users")
    print(f"Hybrid Weights:       CF={config.HYBRID_CF_WEIGHT}, Content={config.HYBRID_CONTENT_WEIGHT}, Weakness={config.HYBRID_WEAKNESS_WEIGHT}")

    print("\nGenerating held-out test population for verification...")
    questions_df = load_canonical_questions()
    if config.CURRENT_PROFILE == "v3":
        users_df, questions_df, subs_df, _ = generate_synthetic_dataset_v3(
            questions_df=questions_df,
            n_users=config.NUM_USERS,
            random_seed=config.RANDOM_SEED,
            zipf_exponent=0.35,
            beta=3.2,
        )
    else:
        users_df, latent_info = generate_synthetic_users(n=config.NUM_USERS, return_latent_info=True)
        subs_df = generate_synthetic_submissions(users_df, questions_df, latent_users_info=latent_info)

    train_users, val_users, test_users, train_subs, val_subs, test_subs = user_train_val_test_split(
        users_df, subs_df, random_seed=config.RANDOM_SEED
    )

    fe_test = FeatureEngineer(test_users, questions_df, test_subs)
    test_matrix = fe_test.build_user_feature_matrix()

    # 1. Rating Predictor
    print("\n" + "-" * 80)
    print("1. CONTEST RATING REGRESSION EVALUATION (Held-out Test Users)")
    print("-" * 80)
    predictor = ContestRatingPredictor()
    predictor.load_model(config.RATING_MODEL_PATH)
    rating_metrics = predictor.evaluate(test_matrix)
    print(f"   RMSE: {rating_metrics['rmse']:.2f} rating points")
    print(f"   MAE:  {rating_metrics['mae']:.2f} rating points")
    print(f"   R^2:  {rating_metrics['r2']:.4f}")

    # 2. Weak-Topic Detection
    print("\n" + "-" * 80)
    print("2. WEAK-TOPIC DETECTION EVALUATION (Predicting Future Failures)")
    print("-" * 80)
    test_hist, test_fut = temporal_split_user_submissions(
        test_subs, history_ratio=config.TEMPORAL_HISTORY_RATIO
    )
    test_uids = test_users["user_id"].tolist()

    weak_metrics = evaluate_weak_topic_detection(
        test_user_ids=test_uids,
        questions_df=questions_df,
        test_history_subs=test_hist,
        test_future_subs=test_fut,
    )
    print(f"   Precision: {weak_metrics.get('precision', 0):.4f}")
    print(f"   Recall:    {weak_metrics.get('recall', 0):.4f}")
    print(f"   F1 Score:  {weak_metrics.get('f1', 0):.4f}")

    # 3. Comprehensive Recommender Baseline Comparison
    print("\n" + "-" * 80)
    print("3. HONEST RECOMMENDER EVALUATION (Overall vs Long-Tail vs Debiased)")
    print("-" * 80)
    als = ALSMatrixFactorization().load(config.ALS_MODEL_PATH)

    rec_results = evaluate_recommenders_comprehensive(
        questions_df=questions_df,
        train_subs=train_subs,
        test_users=test_users,
        test_hist_subs=test_hist,
        test_fut_subs=test_fut,
        als_model=als,
        top_k=config.TOP_N_RECOMMENDATIONS,
        random_seed=config.RANDOM_SEED,
    )

    m = rec_results["methods"]
    n_users_eval = rec_results["metadata"]["evaluated_users"]
    n_tail_users = rec_results["metadata"]["users_with_tail_gt"]
    n_head = rec_results["metadata"]["head_cutoff_top20pct"]

    print(f"Evaluated Test Users: {n_users_eval} (of which {n_tail_users} had held-out long-tail items)")
    print(f"Head/Tail Definition: Top 20% most popular ({n_head} items) vs remaining 80% ({len(questions_df) - n_head} items)")

    # Print Table 1: Overall Recommendations
    print("\n[TABLE 1] OVERALL RECOMMENDATION PERFORMANCE (Top-5 with 95% Bootstrap CI)")
    print(f"{'Method':<28} | {'Precision@5':<12} | {'Recall@5':<10} | {'Hit Rate@5 (95% CI)':<26} | {'NDCG@5 (95% CI)':<24} | {'Tail Share@5':<12}")
    print("-" * 125)
    for name, key in [
        ("Hybrid (Production CF=0.50)", "hybrid"),
        ("Popularity Baseline", "popularity"),
        ("Discovery Mode (CF=0.0)", "discovery"),
        ("Content-Only", "content_only"),
        ("Random Baseline", "random"),
    ]:
        row = m[key]
        hit_str = f"{row['hit_rate@5']*100:.2f}% [{row['hit_rate@5_ci95'][0]*100:.2f}%, {row['hit_rate@5_ci95'][1]*100:.2f}%]"
        ndcg_str = f"{row['ndcg@5']:.4f} [{row['ndcg@5_ci95'][0]:.4f}, {row['ndcg@5_ci95'][1]:.4f}]"
        print(f"{name:<28} | {row['precision@5']:<12.4f} | {row['recall@5']:<10.4f} | {hit_str:<26} | {ndcg_str:<24} | {row['tail_share@5']*100:<11.1f}%")

    diff_data = rec_results.get("paired_differences", {}).get("hybrid_minus_popularity", {})
    if diff_data:
        print(f"\nPaired Differences (Hybrid minus Popularity, 95% Bootstrap CI):")
        print(f"  NDCG@5 Diff:   {diff_data['ndcg@5_diff']:+.4f} (95% CI [{diff_data['ndcg@5_diff_ci95'][0]:+.4f}, {diff_data['ndcg@5_diff_ci95'][1]:+.4f}])")
        print(f"  Hit Rate@5 Diff: {diff_data['hit_rate@5_diff']*100:+.2f}% (95% CI [{diff_data['hit_rate@5_diff_ci95'][0]*100:+.2f}%, {diff_data['hit_rate@5_diff_ci95'][1]*100:+.2f}%])")

    # Print Table 2: Long-Tail Recommendations (Held-out items outside top 20% most popular)
    print("\n[TABLE 2] LONG-TAIL HELD-OUT PERFORMANCE (Items Outside Top 20% Head)")
    print(f"{'Method':<25} | {'Tail Hit@5':<12} | {'Tail Prec@5':<12} | {'Tail Recall@5':<13} | {'Tail NDCG@5':<12}")
    print("-" * 82)
    for name, key in [
        ("Hybrid (Production CF=0.50)", "hybrid"),
        ("Discovery Mode (CF=0.0)", "discovery"),
        ("Popularity Baseline", "popularity"),
        ("Content-Only", "content_only"),
        ("Random Baseline", "random"),
    ]:
        row = m[key]
        print(f"{name:<25} | {row['tail_hit_rate@5']*100:<11.2f}% | {row['tail_precision@5']:<12.4f} | {row['tail_recall@5']:<13.4f} | {row['tail_ndcg@5']:<12.4f}")

    # Print Table 3: Popularity-Debiased Comparison (IPS-NDCG@5)
    print("\n[TABLE 3] POPULARITY-DEBIASED COMPARISON (Inverse Propensity Scored)")
    print(f"{'Method':<25} | {'Standard NDCG@5':<18} | {'Debiased IPS-NDCG@5':<22} | {'Tail Share@5':<12}")
    print("-" * 84)
    for name, key in [
        ("Hybrid (Production CF=0.50)", "hybrid"),
        ("Discovery Mode (CF=0.0)", "discovery"),
        ("Popularity Baseline", "popularity"),
        ("Content-Only", "content_only"),
        ("Random Baseline", "random"),
    ]:
        row = m[key]
        print(f"{name:<25} | {row['ndcg@5']:<18.4f} | {row['ips_ndcg@5']:<22.4f} | {row['tail_share@5']*100:<11.1f}%")

    # Save evaluation summary to results/
    output_path = ROOT_DIR / "results" / "evaluation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "rating_model_test": rating_metrics,
        "weak_topic_test": weak_metrics,
        "recommendation_comparison": rec_results,
    }
    output_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSaved full evaluation summary to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
