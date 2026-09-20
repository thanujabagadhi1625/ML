"""
results/make_tables.py
----------------------
Script to generate all markdown tables for README.md and reporting directly
from JSON files in results/, strictly obeying Global Rule 2:
"Never hand-write or hand-edit any metric. Every number in README.md must be
generated from JSON files in results/ by a script (results/make_tables.py)."
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).resolve().parent


def load_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def make_benchmark_table(data: dict, setting_name: str = "Benchmark") -> str:
    """Generates Markdown table for recommendation benchmark results."""
    methods = ["random", "popularity", "content_only", "als_only", "hybrid"]
    method_labels = {
        "random": "Random",
        "popularity": "Popularity",
        "content_only": "Content-Only",
        "als_only": "ALS-Only",
        "hybrid": "Hybrid Recommender",
    }

    lines = []
    lines.append(f"### {setting_name} Recommendation Results (K=5 & K=10)")
    meta = data.get("metadata", {})
    lines.append(
        f"*Evaluated Users:* {meta.get('evaluated_users_total')} across {meta.get('n_seeds')} seeds "
        f"(mean {meta.get('evaluated_users_mean_per_seed', 0):.1f} users/seed, "
        f"avg ground-truth items: {meta.get('gt_items_per_user_mean', 0):.2f}/user)"
    )
    lines.append(
        f"*ALS Training Time:* {meta.get('als_training_time_mean_seconds', 0):.3f}s | "
        f"*Interactions/user:* {meta.get('interactions_per_user_mean', 0):.1f} | "
        f"*Matrix density:* {meta.get('matrix_density_mean', 0):.4f}\n"
    )

    # Table K=5
    lines.append("#### Metrics at K=5")
    lines.append("| Method | Precision@5 (mean ± std) | 95% Bootstrap CI | Recall@5 | Hit Rate@5 | NDCG@5 | Head Share@5 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    m_dict = data.get("methods", {})
    ht_dict = data.get("head_vs_tail", {})

    for m in methods:
        if m not in m_dict:
            continue
        label = method_labels.get(m, m)
        p5 = m_dict[m].get("precision@5", {})
        r5 = m_dict[m].get("recall@5", {})
        h5 = m_dict[m].get("hit_rate@5", {})
        n5 = m_dict[m].get("ndcg@5", {})
        head5 = ht_dict.get(m, {}).get("head_share@5", {}).get("mean", 0.0)

        p_str = f"{p5.get('mean', 0):.4f} ± {p5.get('std', 0):.4f}"
        ci_str = format_ci(p5.get("ci_95", [0, 0]))
        r_str = f"{r5.get('mean', 0):.4f}"
        h_str = f"{h5.get('mean', 0) * 100:.2f}%"
        n_str = f"{n5.get('mean', 0):.4f}"
        hd_str = f"{head5 * 100:.1f}%"

        lines.append(f"| **{label}** | `{p_str}` | `{ci_str}` | `{r_str}` | `{h_str}` | `{n_str}` | `{hd_str}` |")

    # Paired differences
    pairs = data.get("paired_differences", {})
    if pairs:
        lines.append("\n#### Paired Differences at K=5 (vs Hybrid)")
        lines.append("| Comparison | Diff Precision@5 | 95% CI | Diff Recall@5 | 95% CI | Diff Hit Rate@5 | 95% CI |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for pair_name, pair_label in [
            ("hybrid_minus_popularity", "Hybrid minus Popularity"),
            ("hybrid_minus_random", "Hybrid minus Random"),
        ]:
            if pair_name in pairs:
                dp = pairs[pair_name].get("precision@5", {})
                dr = pairs[pair_name].get("recall@5", {})
                dh = pairs[pair_name].get("hit_rate@5", {})
                lines.append(
                    f"| **{pair_label}** | "
                    f"`{dp.get('mean', 0):+.4f} ± {dp.get('std', 0):.4f}` | `{format_ci(dp.get('ci_95', [0, 0]))}` | "
                    f"`{dr.get('mean', 0):+.4f}` | `{format_ci(dr.get('ci_95', [0, 0]))}` | "
                    f"`{dh.get('mean', 0) * 100:+.2f}%` | `[{dh.get('ci_95', [0, 0])[0]*100:+.2f}%, {dh.get('ci_95', [0, 0])[1]*100:+.2f}%]` |"
                )

    # Table K=10
    lines.append("\n#### Metrics at K=10")
    lines.append("| Method | Precision@10 (mean ± std) | 95% Bootstrap CI | Recall@10 | Hit Rate@10 | NDCG@10 | Head Share@10 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for m in methods:
        if m not in m_dict:
            continue
        label = method_labels.get(m, m)
        p10 = m_dict[m].get("precision@10", {})
        r10 = m_dict[m].get("recall@10", {})
        h10 = m_dict[m].get("hit_rate@10", {})
        n10 = m_dict[m].get("ndcg@10", {})
        head10 = ht_dict.get(m, {}).get("head_share@10", {}).get("mean", 0.0)

        p_str = f"{p10.get('mean', 0):.4f} ± {p10.get('std', 0):.4f}"
        ci_str = format_ci(p10.get("ci_95", [0, 0]))
        r_str = f"{r10.get('mean', 0):.4f}"
        h_str = f"{h10.get('mean', 0) * 100:.2f}%"
        n_str = f"{n10.get('mean', 0):.4f}"
        hd_str = f"{head10 * 100:.1f}%"

        lines.append(f"| **{label}** | `{p_str}` | `{ci_str}` | `{r_str}` | `{h_str}` | `{n_str}` | `{hd_str}` |")

    return "\n".join(lines)


def make_version_history_table(v1_meta: dict | None, v3_meta: dict | None) -> str:
    lines = []
    lines.append("### Generator Version History: v1 vs v3")
    lines.append("| Dimension | Generator v1 (Legacy) | Generator v3 (New Multi-Factor Simulator) |")
    lines.append("| :--- | :--- | :--- |")
    lines.append("| **Topic Interests** | Archetype-level static discrete weights | 20-dim Dirichlet vector per user; archetype shifts concentration mean |")
    lines.append("| **User Skill** | Single static archetype value | Individual Gaussian noise + time-varying chronological evolution |")
    lines.append("| **Question Popularity** | Uniform base choice | Heavy-tailed Zipf power-law distribution |")
    lines.append("| **Choice Mechanics** | Archetype categorical weights | $P(q) \\propto \\text{pop}(q) \\cdot e^{\\beta (\\theta_u \\cdot t_q)} \\cdot \\text{diff\\_fit}(\\text{skill}_u(t), d_q)$ |")
    lines.append("| **Timestamps** | Independent power-law draws; non-sequential | Strictly increasing chronological sequence; real 80/20 temporal split |")
    lines.append("| **Re-attempts** | Static retry loop | Failed submissions retry with $p_{\\text{retry}}$; successes move on |")
    lines.append("| **Submissions Volume** | Fixed global pool allocated by multipliers | Lognormal heavy-tailed per user (min 20, mean ~100) |")
    lines.append("| **Contest Rating Target**| Evaluated on min(age, 365) while history spanned 365d | Consistent: skill evaluated at end of actual submission span |")
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("LEETCODE MENTOR — BENCHMARK RESULTS TABLES (make_tables.py)")
    print("=" * 80)

    # S1 / bench_v1
    bench_v1 = load_json("bench_v1.json") or load_json("s1.json")
    if bench_v1:
        print("\n" + make_benchmark_table(bench_v1, "Setting S1 (Generator v1, 453 Catalogue, 600 Users)"))

    # S2
    s2 = load_json("s2.json")
    if s2:
        print("\n" + make_benchmark_table(s2, "Setting S2 (Generator v3, 453 Catalogue, 600 Users)"))

    # S3
    s3 = load_json("s3.json")
    if s3:
        print("\n" + make_benchmark_table(s3, "Setting S3 (Generator v3, Real Catalogue, 2000 Users)"))
    else:
        print("\n### Setting S3: Not available (requires data/leetcode_problems.csv)")

    # S4
    s4 = load_json("s4.json")
    if s4:
        print("\n" + make_benchmark_table(s4, "Setting S4 (Generator v3, Real Catalogue, 4000 Users)"))
    else:
        print("\n### Setting S4: Not available (requires data/leetcode_problems.csv)")

    # Version history
    print("\n" + make_version_history_table(None, None))


if __name__ == "__main__":
    main()
