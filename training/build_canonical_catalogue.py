"""
training/build_canonical_catalogue.py
------------------------------------
Builds the canonical question catalogue from data/leetcode_problems.csv:
- Drops paid-only questions.
- Maps source topic tags to the 20 canonical tags using an explicit mapping table.
- Drops questions with no mappable tag (never defaults to 'Array').
- Removes ingestion from sync_store.json / demo files / placeholder generated questions.
- Sets description = title when real problem statement text is not present.
- Prints detailed catalogue stats and saves to results/catalogue_stats.json.
- Writes models/canonical_questions.json and regenerates models/question_embeddings.npy.
- Validates that demo file problem slugs map without error.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add files directory to sys.path
FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

import config
from nlp_cluster import WeakTopicAnalyzer

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "leetcode_problems.csv"
CANONICAL_OUTPUT = config.CANONICAL_QUESTIONS_PATH
STATS_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "catalogue_stats.json"

# Explicit source-to-canonical tag mapping table
# Canonical tags: config.TOPIC_TAGS (20 canonical tags)
EXPLICIT_TAG_MAP: Dict[str, str] = {
    # Direct 1-to-1 matches
    "Array": "Array",
    "String": "String",
    "Hash Table": "Hash Table",
    "Dynamic Programming": "Dynamic Programming",
    "Math": "Math",
    "Sorting": "Sorting",
    "Greedy": "Greedy",
    "Depth-First Search": "Depth-First Search",
    "Breadth-First Search": "Breadth-First Search",
    "Binary Search": "Binary Search",
    "Tree": "Tree",
    "Graph": "Graph",
    "Two Pointers": "Two Pointers",
    "Sliding Window": "Sliding Window",
    "Backtracking": "Backtracking",
    "Bit Manipulation": "Bit Manipulation",
    "Heap": "Heap",
    "Stack": "Stack",
    "Trie": "Trie",
    "Union Find": "Union Find",

    # Explicit mappings required by spec
    "Heap (Priority Queue)": "Heap",
    "Binary Tree": "Tree",
    "Binary Search Tree": "Tree",
    "Monotonic Stack": "Stack",
    "Merge Sort": "Sorting",

    # Additional algorithmic variants
    "Quickselect": "Sorting",
    "Bucket Sort": "Sorting",
    "Radix Sort": "Sorting",
    "Counting Sort": "Sorting",
    "Monotonic Queue": "Stack",
    "Disjoint Set": "Union Find",
    "Minimum Spanning Tree": "Graph",
    "Shortest Path": "Graph",
    "Eulerian Circuit": "Graph",
    "Strongly Connected Component": "Graph",
    "Biconnected Component": "Graph",
    "Topological Sort": "Graph",
    "Bitmask": "Bit Manipulation",
    "Combinatorics": "Math",
    "Geometry": "Math",
    "Game Theory": "Math",
    "Probability and Statistics": "Math",
    "Number Theory": "Math",
}


def build_canonical_catalogue() -> list[dict]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Required catalogue dataset not found at '{CSV_PATH}'. "
            "Please ensure data/leetcode_problems.csv exists before running."
        )

    print(f"Reading {CSV_PATH} ...")
    raw_df = pd.read_csv(CSV_PATH)
    rows_read = len(raw_df)
    print(f"1. Rows read: {rows_read}")

    # Drop paidOnly questions
    non_paid_df = raw_df[~raw_df["paidOnly"].astype(bool)].copy()
    rows_after_dropping_paid = len(non_paid_df)
    print(f"2. Rows after dropping paid-only: {rows_after_dropping_paid}")

    # Process and map tags
    catalogue_items = []
    unmapped_tag_counter = Counter()
    dropped_no_mappable_tag = 0

    seen_slugs = set()

    for _, row in non_paid_df.iterrows():
        raw_slug = str(row.get("titleSlug") or "").strip().lower()
        if not raw_slug or raw_slug in seen_slugs:
            continue

        raw_title = str(row.get("title") or raw_slug.replace("-", " ").title()).strip()
        difficulty = str(row.get("difficulty") or "Medium").strip().capitalize()
        if difficulty not in config.DIFFICULTIES:
            difficulty = "Medium"

        # acRate is a percentage in CSV (e.g. 55.32 -> 0.5532)
        raw_ac = row.get("acRate")
        if pd.notna(raw_ac):
            acc_rate = round(float(raw_ac) / 100.0, 4)
        else:
            acc_rate = 0.50

        # Parse tags using ast.literal_eval
        raw_tags_str = row.get("topicTags")
        mapped_tags = []
        if pd.notna(raw_tags_str):
            try:
                parsed_tags = ast.literal_eval(str(raw_tags_str))
                for t in parsed_tags:
                    t_clean = str(t).strip()
                    if t_clean in EXPLICIT_TAG_MAP:
                        canon_tag = EXPLICIT_TAG_MAP[t_clean]
                        if canon_tag not in mapped_tags and canon_tag in config.TOPIC_TAGS:
                            mapped_tags.append(canon_tag)
                    else:
                        unmapped_tag_counter[t_clean] += 1
            except Exception:
                pass

        if not mapped_tags:
            # Dropped because no tags map to canonical 20 tags
            dropped_no_mappable_tag += 1
            continue

        seen_slugs.add(raw_slug)
        catalogue_items.append({
            "slug": raw_slug,
            "title": raw_title,
            "difficulty": difficulty,
            "topic_tags": mapped_tags,
            "acceptance_rate": acc_rate,
            "description": raw_title,  # Real title used as description
        })

    drop_ratio = dropped_no_mappable_tag / max(rows_after_dropping_paid, 1)
    print(f"3. Rows dropped for no mappable tag: {dropped_no_mappable_tag} ({drop_ratio * 100:.2f}%)")

    top_15_unmapped = [
        {"tag": tag, "count": count}
        for tag, count in unmapped_tag_counter.most_common(15)
    ]
    print("\n4. Top 15 most frequent unmapped tags:")
    for item in top_15_unmapped:
        print(f"   - {item['tag']}: {item['count']}")

    if drop_ratio > 0.30:
        raise RuntimeError(
            f"ABORT: {drop_ratio * 100:.2f}% of non-paid questions would be dropped (> 30% limit). "
            f"Dropped {dropped_no_mappable_tag}/{rows_after_dropping_paid}."
        )

    # Assign consistent 1-indexed question_id sorted by slug
    catalogue_items.sort(key=lambda x: x["slug"])
    for qid, item in enumerate(catalogue_items, start=1):
        item["question_id"] = qid

    total_questions = len(catalogue_items)
    difficulty_counts = Counter(item["difficulty"] for item in catalogue_items)
    tag_counts = Counter(tag for item in catalogue_items for tag in item["topic_tags"])
    tags_per_question_list = [len(item["topic_tags"]) for item in catalogue_items]
    tags_per_question_dist = Counter(tags_per_question_list)
    avg_tags_per_question = float(np.mean(tags_per_question_list))

    print(f"\n5. Final catalogue size: {total_questions} questions")
    print(f"   Difficulty distribution: {dict(difficulty_counts)}")
    print(f"   Average tags per question: {avg_tags_per_question:.2f}")

    # Build stats dictionary
    stats = {
        "rows_read": rows_read,
        "rows_after_dropping_paid": rows_after_dropping_paid,
        "rows_dropped_no_mappable_tag": dropped_no_mappable_tag,
        "drop_ratio_percent": round(drop_ratio * 100.0, 2),
        "total_canonical_questions": total_questions,
        "counts_per_difficulty": dict(difficulty_counts),
        "counts_per_canonical_tag": dict(tag_counts),
        "tags_per_question_distribution": {str(k): v for k, v in sorted(tags_per_question_dist.items())},
        "average_tags_per_question": round(avg_tags_per_question, 2),
        "top_15_unmapped_tags": top_15_unmapped,
    }

    STATS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUTPUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved catalogue stats to {STATS_OUTPUT}")

    # Write canonical questions
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_OUTPUT.write_text(json.dumps(catalogue_items, indent=2), encoding="utf-8")
    print(f"Saved {len(catalogue_items)} canonical questions to {CANONICAL_OUTPUT}")

    # Regenerate question embeddings
    print("\nRegenerating models/question_embeddings.npy ...")
    analyzer = WeakTopicAnalyzer()
    descriptions = [item["description"] for item in catalogue_items]
    embeddings = analyzer.embed_descriptions(descriptions)
    np.save(config.QUESTION_EMBEDDINGS_PATH, embeddings)
    print(f"Saved question embeddings of shape {embeddings.shape} to {config.QUESTION_EMBEDDINGS_PATH}")

    # Validate demo file and fold-in mapping
    demo_path = config.FILES_DIR / "demo" / "large_user.json"
    if demo_path.exists():
        demo_data = json.loads(demo_path.read_text(encoding="utf-8"))
        demo_subs = demo_data.get("submissions", [])
        demo_slugs = {s.get("problem_slug") or s.get("question_id") for s in demo_subs}
        cat_slugs = {item["slug"] for item in catalogue_items}
        matched = demo_slugs.intersection(cat_slugs)
        unmatched = demo_slugs - cat_slugs
        print(f"\nDemo File Mapping Verification:")
        print(f"  Total unique demo problem slugs: {len(demo_slugs)}")
        print(f"  Successfully matched in catalogue: {len(matched)}")
        print(f"  Unmatched demo slugs: {len(unmatched)} {unmatched}")
        if unmatched:
            print(f"  Warning: {len(unmatched)} demo slugs unmatched: {unmatched}")
        else:
            print("  All demo slugs successfully mapped to catalogue without errors.")

    return catalogue_items


if __name__ == "__main__":
    build_canonical_catalogue()
