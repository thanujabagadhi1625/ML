"""
training/run_staged_experiments.py
----------------------------------
Executes Staged Experiments S1-S4 and the Sensitivity Study.

Experiments:
  - S1: Generator v1, 453 catalogue (models/v1/canonical_questions.json), 600 users, 10 seeds -> results/s1.json
  - S2: Generator v3, 453 catalogue (models/v1/canonical_questions.json), 600 users, 10 seeds -> results/s2.json
  - S3: Generator v3, real catalogue (models/canonical_questions.json), 2000 users, 10 seeds -> results/s3.json
  - S4: Generator v3, real catalogue (models/canonical_questions.json), 4000 users, 10 seeds -> results/s4.json
  - Sensitivity: S2 setup (453 items, 600 users) across Zipf exponent in [0.0, 0.4, 0.8, 1.2] x beta in [0.5, 1.0, 2.0] -> results/sensitivity.json
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
TRAINING_DIR = ROOT_DIR / "training"
RESULTS_DIR = ROOT_DIR / "results"
for p in [str(FILES_DIR), str(TRAINING_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_recommenders import run_benchmark


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v1_cat_path = ROOT_DIR / "models" / "v1" / "canonical_questions.json"
    real_cat_path = ROOT_DIR / "models" / "canonical_questions.json"

    # Load chosen config if present
    chosen_config_path = RESULTS_DIR / "chosen_config.json"
    if chosen_config_path.exists():
        chosen_data = json.loads(chosen_config_path.read_text(encoding="utf-8"))
        chosen_als = chosen_data.get("chosen_als_params")
        chosen_weights_dict = chosen_data.get("chosen_hybrid_weights", {})
        chosen_weights = (
            chosen_weights_dict.get("cf_weight", 0.5),
            chosen_weights_dict.get("content_weight", 0.3),
            chosen_weights_dict.get("weakness_weight", 0.2),
        )
    else:
        chosen_als = None
        chosen_weights = None

    seeds_10 = list(range(1, 11))

    # --------------------------------------------------------------------------
    # S1: Generator v1, 453 catalogue, 600 users, 10 seeds
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("STARTING EXPERIMENT S1 (Generator v1, 453 catalogue, 600 users)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    run_benchmark(
        seeds=seeds_10,
        k_values=[5, 10],
        generator_version="v1",
        catalogue_path=v1_cat_path,
        n_users=600,
        n_submissions=30_000,
        output_path=RESULTS_DIR / "s1.json",
        verbose=True,
    )
    print(f"Completed S1 in {time.time() - t0:.1f}s\n", flush=True)

    # --------------------------------------------------------------------------
    # S2: Generator v3, 453 catalogue, 600 users, 10 seeds
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("STARTING EXPERIMENT S2 (Generator v3, 453 catalogue, 600 users)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    run_benchmark(
        seeds=seeds_10,
        k_values=[5, 10],
        generator_version="v3",
        catalogue_path=v1_cat_path,
        n_users=600,
        output_path=RESULTS_DIR / "s2.json",
        verbose=True,
    )
    print(f"Completed S2 in {time.time() - t0:.1f}s\n", flush=True)

    # --------------------------------------------------------------------------
    # S3: Generator v3, real catalogue (2630 items), 2000 users, 10 seeds
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("STARTING EXPERIMENT S3 (Generator v3, Real catalogue, 2000 users)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    run_benchmark(
        seeds=seeds_10,
        k_values=[5, 10],
        generator_version="v3",
        catalogue_path=real_cat_path,
        n_users=2000,
        als_params=chosen_als,
        hybrid_weights=chosen_weights,
        output_path=RESULTS_DIR / "s3.json",
        verbose=True,
    )
    print(f"Completed S3 in {time.time() - t0:.1f}s\n", flush=True)

    # --------------------------------------------------------------------------
    # S4: Generator v3, real catalogue (2630 items), 4000 users, 10 seeds
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("STARTING EXPERIMENT S4 (Generator v3, Real catalogue, 4000 users)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    run_benchmark(
        seeds=seeds_10,
        k_values=[5, 10],
        generator_version="v3",
        catalogue_path=real_cat_path,
        n_users=4000,
        als_params=chosen_als,
        hybrid_weights=chosen_weights,
        output_path=RESULTS_DIR / "s4.json",
        verbose=True,
    )
    print(f"Completed S4 in {time.time() - t0:.1f}s\n", flush=True)

    # --------------------------------------------------------------------------
    # Sensitivity Study: S2 setup x Zipf [0.0, 0.4, 0.8, 1.2] x beta [0.5, 1.0, 2.0]
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("STARTING SENSITIVITY STUDY (S2 setup: Zipf exponent x beta)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    zipf_values = [0.0, 0.4, 0.8, 1.2]
    beta_values = [0.5, 1.0, 2.0]
    sensitivity_seeds = [1, 2, 3]  # Multi-seed evaluation for sensitivity grid

    grid_results = []
    total_runs = len(zipf_values) * len(beta_values)
    run_count = 0

    for z_exp, b_val in itertools.product(zipf_values, beta_values):
        run_count += 1
        print(f"\n--- Sensitivity [{run_count}/{total_runs}]: zipf={z_exp}, beta={b_val} ---", flush=True)
        res = run_benchmark(
            seeds=sensitivity_seeds,
            k_values=[5, 10],
            generator_version="v3",
            catalogue_path=v1_cat_path,
            n_users=600,
            zipf_exponent=z_exp,
            beta=b_val,
            verbose=False,
        )

        entry = {
            "zipf_exponent": z_exp,
            "beta": b_val,
            "n_seeds": len(sensitivity_seeds),
            "seeds": sensitivity_seeds,
            "popularity": {
                "precision@5": res["methods"]["popularity"]["precision@5"],
                "hit_rate@5": res["methods"]["popularity"]["hit_rate@5"],
                "ndcg@5": res["methods"]["popularity"]["ndcg@5"],
                "head_share@5": res["head_vs_tail"]["popularity"]["head_share@5"],
            },
            "hybrid": {
                "precision@5": res["methods"]["hybrid"]["precision@5"],
                "hit_rate@5": res["methods"]["hybrid"]["hit_rate@5"],
                "ndcg@5": res["methods"]["hybrid"]["ndcg@5"],
                "head_share@5": res["head_vs_tail"]["hybrid"]["head_share@5"],
                "weak_coverage@5": res["methods"]["hybrid"]["weak_coverage@5"],
                "difficulty_fit@5": res["methods"]["hybrid"]["difficulty_fit@5"],
            },
            "als_only": {
                "precision@5": res["methods"]["als_only"]["precision@5"],
                "hit_rate@5": res["methods"]["als_only"]["hit_rate@5"],
                "ndcg@5": res["methods"]["als_only"]["ndcg@5"],
            },
            "random": {
                "precision@5": res["methods"]["random"]["precision@5"],
                "hit_rate@5": res["methods"]["random"]["hit_rate@5"],
                "ndcg@5": res["methods"]["random"]["ndcg@5"],
                "head_share@5": res["head_vs_tail"]["random"]["head_share@5"],
            },
            "paired_diff_hybrid_minus_popularity_p@5": res["paired_differences"]["hybrid_minus_popularity"]["precision@5"],
            "paired_diff_hybrid_minus_als_only_p@5": res["paired_differences"]["hybrid_minus_als_only"]["precision@5"],
        }
        grid_results.append(entry)
        print(
            f"  Result: Pop P@5={entry['popularity']['precision@5']['mean']:.4f}, "
            f"Hybrid P@5={entry['hybrid']['precision@5']['mean']:.4f}, "
            f"ALS P@5={entry['als_only']['precision@5']['mean']:.4f}, "
            f"Diff(H-Pop)={entry['paired_diff_hybrid_minus_popularity_p@5']['mean']:+.4f}",
            flush=True,
        )

    sens_output = {
        "metadata": {
            "experiment": "Sensitivity Study: Zipf Exponent x Beta on S2 setup",
            "catalogue": "453 questions (models/v1/canonical_questions.json)",
            "n_users": 600,
            "seeds": sensitivity_seeds,
            "generator_version": "v3",
            "zipf_values": zipf_values,
            "beta_values": beta_values,
        },
        "grid": grid_results,
    }

    sens_path = RESULTS_DIR / "sensitivity.json"
    sens_path.write_text(json.dumps(sens_output, indent=2), encoding="utf-8")
    print(f"\nSaved sensitivity study results to {sens_path} (Total time: {time.time() - t0:.1f}s)", flush=True)
    print("\nALL STAGED EXPERIMENTS S1-S4 AND SENSITIVITY STUDY COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
