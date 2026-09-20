"""
training/tune_recommendations.py
--------------------------------
Systematic hyperparameter tuning on VALIDATION users only (seeds 1-3).
Strictly obeys Global Rule 3:
  "Never tune anything on test users. Tune only on validation users."

Grid search:
  1. ALS parameters:
     - factors: [16, 24, 48]
     - alpha: [5, 15, 40]
     - reg_lambda: [5, 20, 50]
     Metric: validation NDCG@10 (mean across seeds 1-3)
  2. Recommendation blend weights:
     - (w_cf, w_content, w_weak) in steps of 0.1 summing to 1.0 (66 combinations)
     Metric: validation NDCG@10 (mean across seeds 1-3)

Outputs:
  - results/tuning.json (full grid search logs)
  - results/chosen_config.json (winning parameters + one-time test evaluation)
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
TRAINING_DIR = ROOT_DIR / "training"
for p in [str(FILES_DIR), str(TRAINING_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from data_processing import (
    compute_user_topic_profile,
    generate_synthetic_dataset_v3,
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


def compute_ndcg_at_k(recommended_qids: list[int], gt_qids: set[int], k: int = 10) -> float:
    top_k = recommended_qids[:k]
    matched = [1 if q in gt_qids else 0 for q in top_k]
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(matched))
    ideal_hits = min(k, len(gt_qids))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def min_max_norm(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


def run_tuning(
    seeds: list[int] = [1, 2, 3],
    n_users: int = 2000,
    output_tuning_path: str | Path = ROOT_DIR / "results" / "tuning.json",
    output_chosen_path: str | Path = ROOT_DIR / "results" / "chosen_config.json",
    verbose: bool = True,
) -> dict[str, Any]:
    print("=" * 80, flush=True)
    print("STAGE 4: VALIDATION TUNING ON VALIDATION USERS ONLY (SEEDS 1-3)", flush=True)
    print("=" * 80, flush=True)

    questions_df = load_canonical_questions().reset_index(drop=True)
    all_qids = questions_df["question_id"].tolist()
    qid_to_idx = {q: i for i, q in enumerate(all_qids)}
    qid_to_primary_tag = {
        row["question_id"]: (row["topic_tags"][0] if row["topic_tags"] else "General")
        for _, row in questions_df.iterrows()
    }

    content_model = ContentBasedRecommender(questions_df)

    fallback_acc = questions_df["acceptance_rate"].fillna(0.5).to_numpy()
    diff_score = np.array([
        1.0 if d == "Easy" else 0.65 if d == "Medium" else 0.35
        for d in questions_df["difficulty"]
    ])
    fallback_blended = 0.5 * min_max_norm(fallback_acc) + 0.5 * min_max_norm(diff_score)

    # 1. Generate data for seeds 1-3 and prepare train and val splits
    seed_data = {}
    for seed in seeds:
        if verbose:
            print(f"Generating synthetic dataset (v3) for seed {seed}...", flush=True)
        u_df, q_df, s_df, _ = generate_synthetic_dataset_v3(
            questions_df=questions_df,
            n_users=n_users,
            random_seed=seed,
        )
        train_u, val_u, test_u, train_s, val_s, test_s = user_train_val_test_split(
            u_df, s_df, random_seed=seed
        )
        val_h, val_f = temporal_split_user_submissions(
            val_s, history_ratio=config.TEMPORAL_HISTORY_RATIO
        )
        val_solved_f = (
            val_f[val_f["status"] == "Accepted"]
            .groupby("user_id")["question_id"]
            .apply(set)
            .to_dict()
        )
        val_solved_h = (
            val_h[val_h["status"] == "Accepted"]
            .groupby("user_id")["question_id"]
            .apply(set)
            .to_dict()
        )

        train_interactions = build_interaction_matrix(train_s)
        train_solved = train_s[train_s["status"] == "Accepted"]
        pop_counts = train_solved.groupby("question_id")["user_id"].nunique().to_dict()
        pop_ordered = sorted(all_qids, key=lambda q: (-pop_counts.get(q, 0), q))

        eval_users = []
        for uid in sorted(val_u["user_id"].unique()):
            past_s = val_solved_h.get(uid, set())
            fut_s = val_solved_f.get(uid, set())
            gt = fut_s - past_s
            if not gt:
                continue
            u_hist = val_h[val_h["user_id"] == uid]

            cand_indices = np.array([i for i, q in enumerate(all_qids) if q not in past_s], dtype=int)
            cand_qids = [all_qids[i] for i in cand_indices]

            # Content scores
            solved_ids = list(past_s)
            failed_ids = u_hist[u_hist["status"] != "Accepted"]["question_id"].unique().tolist()
            cnt_scores = content_model.similar_to_user_profile(solved_ids, failed_ids)
            cnt_norm = min_max_norm(cnt_scores)

            # Weakness scores
            u_topic_profile = compute_user_topic_profile(u_hist, questions_df, user_id=uid)
            weakness_boost = np.zeros(len(questions_df))
            if u_topic_profile is not None and not u_topic_profile.empty:
                weak_map = u_topic_profile.set_index("topic")["risk_score"].to_dict()
                for idx, tags in enumerate(questions_df["topic_tags"]):
                    tag_boost = max([weak_map.get(tag, 0.0) for tag in tags] + [0.0])
                    weakness_boost[idx] = tag_boost
            wk_norm = min_max_norm(weakness_boost)

            # Precompute interaction matrix once per user
            if not u_hist.empty:
                inter = build_interaction_matrix(u_hist)
                inter_qids = inter["question_id"].tolist()
                inter_weights = inter["weight"].tolist()
            else:
                inter_qids = []
                inter_weights = []

            pop_recs = [q for q in pop_ordered if q not in past_s]

            eval_users.append({
                "uid": uid,
                "past_solved": past_s,
                "gt_solved": gt,
                "cand_indices": cand_indices,
                "cand_qids": cand_qids,
                "content_norm": cnt_norm,
                "weakness_norm": wk_norm,
                "inter_qids": inter_qids,
                "inter_weights": inter_weights,
                "pop_recs": pop_recs,
            })

        seed_data[seed] = {
            "train_interactions": train_interactions,
            "eval_users": eval_users,
            "pop_ordered": pop_ordered,
            "test_u": test_u,
            "test_s": test_s,
        }

    # --------------------------------------------------------------------------
    # 2. Grid Search ALS Hyperparameters: factors x alpha x lambda
    # --------------------------------------------------------------------------
    factor_candidates = [16, 24, 48]
    alpha_candidates = [5.0, 15.0, 40.0]
    lambda_candidates = [5.0, 20.0, 50.0]

    als_grid = []
    best_als_ndcg = -1.0
    best_als_params = None

    total_als_combos = len(factor_candidates) * len(alpha_candidates) * len(lambda_candidates)
    if verbose:
        print(f"\nEvaluating {total_als_combos} ALS parameter combinations on validation users...", flush=True)

    combo_idx = 0
    t0_als = time.perf_counter()
    for f, a, l in itertools.product(factor_candidates, alpha_candidates, lambda_candidates):
        combo_idx += 1
        seed_ndcgs = []

        for seed in seeds:
            data = seed_data[seed]
            als = ALSMatrixFactorization(
                n_factors=f,
                alpha=a,
                reg_lambda=l,
                n_epochs=15,
                random_state=seed,
            ).fit(data["train_interactions"])

            item_ids = als.item_ids_ordered()
            item_id_to_idx = {qid: i for i, qid in enumerate(item_ids)}

            user_ndcgs = []
            for u in data["eval_users"]:
                if u["inter_qids"]:
                    raw_cf, cf_ok = als.fold_in_user(u["inter_qids"], u["inter_weights"])
                    if cf_ok:
                        cf_by_qid = dict(zip(item_ids, raw_cf))
                        cand_scores = [cf_by_qid.get(q, 0.0) for q in u["cand_qids"]]
                        order = np.argsort(-np.array(cand_scores), kind="mergesort")
                        recs = [u["cand_qids"][i] for i in order]
                    else:
                        recs = u["pop_recs"]
                else:
                    recs = u["pop_recs"]

                ndcg = compute_ndcg_at_k(recs, u["gt_solved"], k=10)
                user_ndcgs.append(ndcg)

            seed_ndcgs.append(float(np.mean(user_ndcgs)) if user_ndcgs else 0.0)

        mean_ndcg = float(np.mean(seed_ndcgs))
        std_ndcg = float(np.std(seed_ndcgs, ddof=1)) if len(seed_ndcgs) > 1 else 0.0

        grid_entry = {
            "n_factors": f,
            "alpha": a,
            "reg_lambda": l,
            "val_ndcg@10_mean": round(mean_ndcg, 5),
            "val_ndcg@10_std": round(std_ndcg, 5),
        }
        als_grid.append(grid_entry)

        if verbose:
            print(f"  [{combo_idx:02d}/{total_als_combos}] factors={f:2d}, alpha={a:4.1f}, lambda={l:4.1f} -> val NDCG@10 = {mean_ndcg:.5f} ± {std_ndcg:.5f}", flush=True)

        if mean_ndcg > best_als_ndcg:
            best_als_ndcg = mean_ndcg
            best_als_params = {"n_factors": f, "alpha": a, "reg_lambda": l, "n_epochs": 15}

    print(f"\nALS grid completed in {time.perf_counter() - t0_als:.1f}s", flush=True)
    print(f"Best ALS parameters: {best_als_params} (val NDCG@10 = {best_als_ndcg:.5f})", flush=True)

    # --------------------------------------------------------------------------
    # 3. Grid Search Recommendation Blend Weights
    # --------------------------------------------------------------------------
    if verbose:
        print("\nPrecomputing ALS fold-in scores for best ALS model...", flush=True)

    for seed in seeds:
        data = seed_data[seed]
        als = ALSMatrixFactorization(
            n_factors=best_als_params["n_factors"],
            alpha=best_als_params["alpha"],
            reg_lambda=best_als_params["reg_lambda"],
            n_epochs=15,
            random_state=seed,
        ).fit(data["train_interactions"])
        data["best_als"] = als

        item_ids = als.item_ids_ordered()
        for u in data["eval_users"]:
            cf_available = False
            cf_scores = np.zeros(len(questions_df))
            if u["inter_qids"]:
                raw_cf, cf_available = als.fold_in_user(u["inter_qids"], u["inter_weights"])
                if cf_available:
                    cf_dict = dict(zip(item_ids, raw_cf))
                    cf_scores = questions_df["question_id"].map(cf_dict).fillna(0.0).to_numpy()

            u["cf_available"] = cf_available
            u["cf_norm"] = min_max_norm(cf_scores)

    # Generate blend weight tuples summing to 1.0 in steps of 0.1
    weight_tuples = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            weight_tuples.append((round(i / 10.0, 1), round(j / 10.0, 1), round(k / 10.0, 1)))

    if verbose:
        print(f"Evaluating {len(weight_tuples)} blend weight combinations on validation users...", flush=True)

    weights_grid = []
    best_weights_ndcg = -1.0
    best_weights = None

    t0_weights = time.perf_counter()
    for w_cf, w_cnt, w_wk in weight_tuples:
        seed_ndcgs = []

        for seed in seeds:
            data = seed_data[seed]
            user_ndcgs = []

            for u in data["eval_users"]:
                cf_norm = u["cf_norm"]
                cnt_norm = u["content_norm"]
                wk_norm = u["weakness_norm"]

                if u["cf_available"]:
                    blended = w_cf * cf_norm + w_cnt * cnt_norm + w_wk * wk_norm
                else:
                    blended = (w_cf + w_cnt) * cnt_norm + w_wk * wk_norm

                if not np.any(blended > 0):
                    blended = fallback_blended

                cand_indices = u["cand_indices"]
                cand_qids = u["cand_qids"]
                cand_scores = blended[cand_indices]

                # Stable sort descending
                order = np.argsort(-cand_scores, kind="mergesort")
                cands_sorted = [cand_qids[i] for i in order]

                # Apply diversity constraint (max 2 per primary tag)
                selected = []
                topic_counts: dict[str, int] = {}
                for qid in cands_sorted:
                    tag = qid_to_primary_tag.get(qid, "General")
                    if topic_counts.get(tag, 0) >= 2 and len(selected) < 9:
                        continue
                    selected.append(qid)
                    topic_counts[tag] = topic_counts.get(tag, 0) + 1
                    if len(selected) >= 10:
                        break

                if len(selected) < 10:
                    for qid in cands_sorted:
                        if qid not in selected:
                            selected.append(qid)
                            if len(selected) >= 10:
                                break

                ndcg = compute_ndcg_at_k(selected, u["gt_solved"], k=10)
                user_ndcgs.append(ndcg)

            seed_ndcgs.append(float(np.mean(user_ndcgs)) if user_ndcgs else 0.0)

        mean_ndcg = float(np.mean(seed_ndcgs))
        std_ndcg = float(np.std(seed_ndcgs, ddof=1)) if len(seed_ndcgs) > 1 else 0.0

        grid_entry = {
            "weights": [w_cf, w_cnt, w_wk],
            "val_ndcg@10_mean": round(mean_ndcg, 5),
            "val_ndcg@10_std": round(std_ndcg, 5),
        }
        weights_grid.append(grid_entry)

        if mean_ndcg > best_weights_ndcg:
            best_weights_ndcg = mean_ndcg
            best_weights = [w_cf, w_cnt, w_wk]

    print(f"Weights grid completed in {time.perf_counter() - t0_weights:.1f}s", flush=True)
    if verbose:
        print(f"\nBest Blend Weights: CF={best_weights[0]}, Content={best_weights[1]}, Weakness={best_weights[2]} (val NDCG@10 = {best_weights_ndcg:.5f})", flush=True)

    # --------------------------------------------------------------------------
    # 4. Save results/tuning.json
    # --------------------------------------------------------------------------
    tuning_output = {
        "metadata": {
            "seeds": seeds,
            "n_users": n_users,
            "generator_version": "v3",
            "evaluated_split": "validation_users_only",
            "metric": "val_ndcg@10",
        },
        "best_als_params": best_als_params,
        "best_als_ndcg@10": round(best_als_ndcg, 5),
        "best_weights": {
            "cf_weight": best_weights[0],
            "content_weight": best_weights[1],
            "weakness_weight": best_weights[2],
        },
        "best_weights_ndcg@10": round(best_weights_ndcg, 5),
        "als_grid": sorted(als_grid, key=lambda x: -x["val_ndcg@10_mean"]),
        "weights_grid": sorted(weights_grid, key=lambda x: -x["val_ndcg@10_mean"]),
    }

    Path(output_tuning_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_tuning_path).write_text(json.dumps(tuning_output, indent=2), encoding="utf-8")
    print(f"Saved tuning results to {output_tuning_path}", flush=True)

    # --------------------------------------------------------------------------
    # 5. One-time Test Evaluation of Chosen Setting on TEST Users (Seeds 1-3)
    # --------------------------------------------------------------------------
    if verbose:
        print("\nEvaluating chosen setting once on TEST users (seeds 1-3)...", flush=True)

    from benchmark_recommenders import run_benchmark
    test_bench = run_benchmark(
        seeds=seeds,
        k_values=[5, 10],
        generator_version="v3",
        n_users=n_users,
        als_params=best_als_params,
        hybrid_weights=tuple(best_weights),
        evaluate_on="test",
        verbose=False,
    )

    chosen_config_output = {
        "chosen_als_params": best_als_params,
        "chosen_hybrid_weights": {
            "cf_weight": best_weights[0],
            "content_weight": best_weights[1],
            "weakness_weight": best_weights[2],
        },
        "validation_tuning_summary": {
            "val_ndcg@10_als_only": round(best_als_ndcg, 5),
            "val_ndcg@10_hybrid": round(best_weights_ndcg, 5),
            "seeds_used": seeds,
            "split": "validation_users_only",
        },
        "test_evaluation_summary": {
            "evaluated_users_total": test_bench["metadata"]["evaluated_users_total"],
            "hybrid_precision@5": test_bench["methods"]["hybrid"]["precision@5"],
            "hybrid_ndcg@10": test_bench["methods"]["hybrid"]["ndcg@10"],
            "hybrid_weak_coverage@5": test_bench["methods"]["hybrid"]["weak_coverage@5"],
            "hybrid_difficulty_fit@5": test_bench["methods"]["hybrid"]["difficulty_fit@5"],
            "popularity_precision@5": test_bench["methods"]["popularity"]["precision@5"],
            "als_only_precision@5": test_bench["methods"]["als_only"]["precision@5"],
            "hybrid_minus_popularity_p@5": test_bench["paired_differences"]["hybrid_minus_popularity"]["precision@5"],
            "hybrid_minus_als_only_p@5": test_bench["paired_differences"]["hybrid_minus_als_only"]["precision@5"],
        },
    }

    Path(output_chosen_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_chosen_path).write_text(json.dumps(chosen_config_output, indent=2), encoding="utf-8")
    print(f"Saved chosen config and test evaluation to {output_chosen_path}", flush=True)

    return tuning_output


if __name__ == "__main__":
    run_tuning(seeds=[1, 2, 3], n_users=2000, verbose=True)
