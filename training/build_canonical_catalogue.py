"""
build_canonical_catalogue.py
----------------------------
Builds the single source of truth canonical question catalogue (`models/canonical_questions.json`).
Integrates real problems from sync_store / demo datasets with standard LeetCode reference
problems covering all 20 canonical topic tags across Easy, Medium, and Hard difficulties.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Add files directory to sys.path
FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

import config
from data_processing import _infer_topic_tags_from_slug_or_title, _parse_topic_tags

CANONICAL_OUTPUT = config.CANONICAL_QUESTIONS_PATH

# Well-known reference LeetCode questions to ensure rich coverage across every topic & difficulty
REFERENCE_QUESTIONS = [
    # Dynamic Programming
    {"slug": "climbing-stairs", "title": "Climbing Stairs", "difficulty": "Easy", "topic_tags": ["Dynamic Programming", "Math"], "acceptance_rate": 0.52},
    {"slug": "coin-change", "title": "Coin Change", "difficulty": "Medium", "topic_tags": ["Dynamic Programming", "Breadth-First Search"], "acceptance_rate": 0.43},
    {"slug": "longest-increasing-subsequence", "title": "Longest Increasing Subsequence", "difficulty": "Medium", "topic_tags": ["Dynamic Programming", "Binary Search", "Array"], "acceptance_rate": 0.54},
    {"slug": "edit-distance", "title": "Edit Distance", "difficulty": "Hard", "topic_tags": ["Dynamic Programming", "String"], "acceptance_rate": 0.55},
    {"slug": "trapping-rain-water", "title": "Trapping Rain Water", "difficulty": "Hard", "topic_tags": ["Two Pointers", "Dynamic Programming", "Stack"], "acceptance_rate": 0.60},
    {"slug": "maximum-subarray", "title": "Maximum Subarray", "difficulty": "Medium", "topic_tags": ["Array", "Dynamic Programming"], "acceptance_rate": 0.51},
    {"slug": "house-robber", "title": "House Robber", "difficulty": "Medium", "topic_tags": ["Array", "Dynamic Programming"], "acceptance_rate": 0.50},
    {"slug": "house-robber-ii", "title": "House Robber II", "difficulty": "Medium", "topic_tags": ["Array", "Dynamic Programming"], "acceptance_rate": 0.42},
    {"slug": "word-break", "title": "Word Break", "difficulty": "Medium", "topic_tags": ["Dynamic Programming", "Hash Table", "Trie"], "acceptance_rate": 0.46},
    {"slug": "partition-equal-subset-sum", "title": "Partition Equal Subset Sum", "difficulty": "Medium", "topic_tags": ["Dynamic Programming", "Array"], "acceptance_rate": 0.47},
    {"slug": "burst-balloons", "title": "Burst Balloons", "difficulty": "Hard", "topic_tags": ["Dynamic Programming", "Array"], "acceptance_rate": 0.58},

    # Graph, DFS, BFS
    {"slug": "course-schedule", "title": "Course Schedule", "difficulty": "Medium", "topic_tags": ["Graph", "Depth-First Search", "Breadth-First Search"], "acceptance_rate": 0.47},
    {"slug": "course-schedule-ii", "title": "Course Schedule II", "difficulty": "Medium", "topic_tags": ["Graph", "Depth-First Search", "Breadth-First Search"], "acceptance_rate": 0.50},
    {"slug": "number-of-islands", "title": "Number of Islands", "difficulty": "Medium", "topic_tags": ["Graph", "Breadth-First Search", "Depth-First Search", "Union Find"], "acceptance_rate": 0.58},
    {"slug": "clone-graph", "title": "Clone Graph", "difficulty": "Medium", "topic_tags": ["Hash Table", "Depth-First Search", "Breadth-First Search", "Graph"], "acceptance_rate": 0.55},
    {"slug": "word-ladder", "title": "Word Ladder", "difficulty": "Hard", "topic_tags": ["Hash Table", "String", "Breadth-First Search"], "acceptance_rate": 0.38},
    {"slug": "network-delay-time", "title": "Network Delay Time", "difficulty": "Medium", "topic_tags": ["Depth-First Search", "Breadth-First Search", "Graph", "Heap"], "acceptance_rate": 0.53},
    {"slug": "cheapest-flights-within-k-stops", "title": "Cheapest Flights Within K Stops", "difficulty": "Medium", "topic_tags": ["Dynamic Programming", "Depth-First Search", "Breadth-First Search", "Graph"], "acceptance_rate": 0.38},
    {"slug": "alien-dictionary", "title": "Alien Dictionary", "difficulty": "Hard", "topic_tags": ["Array", "String", "Depth-First Search", "Breadth-First Search", "Graph"], "acceptance_rate": 0.35},

    # Trees, BST
    {"slug": "maximum-depth-of-binary-tree", "title": "Maximum Depth of Binary Tree", "difficulty": "Easy", "topic_tags": ["Tree", "Depth-First Search", "Breadth-First Search"], "acceptance_rate": 0.75},
    {"slug": "invert-binary-tree", "title": "Invert Binary Tree", "difficulty": "Easy", "topic_tags": ["Tree", "Depth-First Search", "Breadth-First Search"], "acceptance_rate": 0.77},
    {"slug": "validate-binary-search-tree", "title": "Validate Binary Search Tree", "difficulty": "Medium", "topic_tags": ["Tree", "Depth-First Search", "Binary Search"], "acceptance_rate": 0.33},
    {"slug": "lowest-common-ancestor-of-a-binary-tree", "title": "Lowest Common Ancestor of a Binary Tree", "difficulty": "Medium", "topic_tags": ["Tree", "Depth-First Search"], "acceptance_rate": 0.61},
    {"slug": "binary-tree-maximum-path-sum", "title": "Binary Tree Maximum Path Sum", "difficulty": "Hard", "topic_tags": ["Dynamic Programming", "Tree", "Depth-First Search"], "acceptance_rate": 0.40},
    {"slug": "serialize-and-deserialize-binary-tree", "title": "Serialize and Deserialize Binary Tree", "difficulty": "Hard", "topic_tags": ["String", "Tree", "Depth-First Search", "Breadth-First Search"], "acceptance_rate": 0.56},

    # Backtracking
    {"slug": "subsets", "title": "Subsets", "difficulty": "Medium", "topic_tags": ["Array", "Backtracking", "Bit Manipulation"], "acceptance_rate": 0.77},
    {"slug": "permutations", "title": "Permutations", "difficulty": "Medium", "topic_tags": ["Array", "Backtracking"], "acceptance_rate": 0.78},
    {"slug": "combination-sum", "title": "Combination Sum", "difficulty": "Medium", "topic_tags": ["Array", "Backtracking"], "acceptance_rate": 0.70},
    {"slug": "generate-parentheses", "title": "Generate Parentheses", "difficulty": "Medium", "topic_tags": ["String", "Dynamic Programming", "Backtracking"], "acceptance_rate": 0.74},
    {"slug": "n-queens", "title": "N-Queens", "difficulty": "Hard", "topic_tags": ["Array", "Backtracking"], "acceptance_rate": 0.67},
    {"slug": "word-search", "title": "Word Search", "difficulty": "Medium", "topic_tags": ["Array", "Backtracking", "Matrix"], "acceptance_rate": 0.42},
    {"slug": "sudoku-solver", "title": "Sudoku Solver", "difficulty": "Hard", "topic_tags": ["Array", "Hash Table", "Backtracking", "Matrix"], "acceptance_rate": 0.60},

    # Sliding Window & Two Pointers
    {"slug": "longest-substring-without-repeating-characters", "title": "Longest Substring Without Repeating Characters", "difficulty": "Medium", "topic_tags": ["Hash Table", "String", "Sliding Window"], "acceptance_rate": 0.35},
    {"slug": "minimum-window-substring", "title": "Minimum Window Substring", "difficulty": "Hard", "topic_tags": ["Hash Table", "String", "Sliding Window"], "acceptance_rate": 0.42},
    {"slug": "sliding-window-maximum", "title": "Sliding Window Maximum", "difficulty": "Hard", "topic_tags": ["Array", "Queue", "Sliding Window", "Heap"], "acceptance_rate": 0.47},
    {"slug": "container-with-most-water", "title": "Container With Most Water", "difficulty": "Medium", "topic_tags": ["Array", "Two Pointers", "Greedy"], "acceptance_rate": 0.55},
    {"slug": "two-sum", "title": "Two Sum", "difficulty": "Easy", "topic_tags": ["Array", "Hash Table"], "acceptance_rate": 0.51},
    {"slug": "3sum", "title": "3Sum", "difficulty": "Medium", "topic_tags": ["Array", "Two Pointers", "Sorting"], "acceptance_rate": 0.34},

    # Binary Search
    {"slug": "binary-search", "title": "Binary Search", "difficulty": "Easy", "topic_tags": ["Array", "Binary Search"], "acceptance_rate": 0.57},
    {"slug": "search-in-rotated-sorted-array", "title": "Search in Rotated Sorted Array", "difficulty": "Medium", "topic_tags": ["Array", "Binary Search"], "acceptance_rate": 0.40},
    {"slug": "find-minimum-in-rotated-sorted-array", "title": "Find Minimum in Rotated Sorted Array", "difficulty": "Medium", "topic_tags": ["Array", "Binary Search"], "acceptance_rate": 0.50},
    {"slug": "median-of-two-sorted-arrays", "title": "Median of Two Sorted Arrays", "difficulty": "Hard", "topic_tags": ["Array", "Binary Search"], "acceptance_rate": 0.39},

    # Heap & Stack
    {"slug": "kth-largest-element-in-an-array", "title": "Kth Largest Element in an Array", "difficulty": "Medium", "topic_tags": ["Array", "Sorting", "Heap"], "acceptance_rate": 0.67},
    {"slug": "merge-k-sorted-lists", "title": "Merge k Sorted Lists", "difficulty": "Hard", "topic_tags": ["Linked List", "Heap", "Divide and Conquer"], "acceptance_rate": 0.52},
    {"slug": "find-median-from-data-stream", "title": "Find Median from Data Stream", "difficulty": "Hard", "topic_tags": ["Two Pointers", "Sorting", "Heap"], "acceptance_rate": 0.52},
    {"slug": "valid-parentheses", "title": "Valid Parentheses", "difficulty": "Easy", "topic_tags": ["String", "Stack"], "acceptance_rate": 0.41},
    {"slug": "daily-temperatures", "title": "Daily Temperatures", "difficulty": "Medium", "topic_tags": ["Array", "Stack", "Monotonic Stack"], "acceptance_rate": 0.66},
    {"slug": "largest-rectangle-in-histogram", "title": "Largest Rectangle in Histogram", "difficulty": "Hard", "topic_tags": ["Array", "Stack", "Monotonic Stack"], "acceptance_rate": 0.44},

    # Trie & Union Find
    {"slug": "implement-trie-prefix-tree", "title": "Implement Trie (Prefix Tree)", "difficulty": "Medium", "topic_tags": ["Hash Table", "String", "Trie"], "acceptance_rate": 0.64},
    {"slug": "word-search-ii", "title": "Word Search II", "difficulty": "Hard", "topic_tags": ["Array", "String", "Backtracking", "Trie"], "acceptance_rate": 0.36},
    {"slug": "redundant-connection", "title": "Redundant Connection", "difficulty": "Medium", "topic_tags": ["Depth-First Search", "Breadth-First Search", "Union Find", "Graph"], "acceptance_rate": 0.63},
    {"slug": "number-of-provinces", "title": "Number of Provinces", "difficulty": "Medium", "topic_tags": ["Depth-First Search", "Breadth-First Search", "Union Find", "Graph"], "acceptance_rate": 0.66},

    # Bit Manipulation & Math
    {"slug": "number-of-1-bits", "title": "Number of 1 Bits", "difficulty": "Easy", "topic_tags": ["Bit Manipulation", "Math"], "acceptance_rate": 0.70},
    {"slug": "counting-bits", "title": "Counting Bits", "difficulty": "Easy", "topic_tags": ["Dynamic Programming", "Bit Manipulation"], "acceptance_rate": 0.78},
    {"slug": "reverse-bits", "title": "Reverse Bits", "difficulty": "Easy", "topic_tags": ["Bit Manipulation", "Divide and Conquer"], "acceptance_rate": 0.58},
    {"slug": "single-number", "title": "Single Number", "difficulty": "Easy", "topic_tags": ["Array", "Bit Manipulation"], "acceptance_rate": 0.72},
    {"slug": "sum-of-two-integers", "title": "Sum of Two Integers", "difficulty": "Medium", "topic_tags": ["Math", "Bit Manipulation"], "acceptance_rate": 0.51},
]


def _generate_problem_description(title: str, difficulty: str, topic_tags: list[str]) -> str:
    tags_str = ", ".join(topic_tags) if topic_tags else "algorithms"
    diff_lower = difficulty.lower()
    return (
        f"Given a problem instance for '{title}', design an optimal {diff_lower}-difficulty algorithm "
        f"leveraging {tags_str}. The solution must optimize runtime complexity within platform memory bounds "
        f"and handle edge cases such as empty inputs, boundary constraints, and duplicate elements."
    )


def build_canonical_catalogue() -> list[dict]:
    catalogue_map: dict[str, dict] = {}

    def add_question(slug: str, title: str, difficulty: str, tags: list[str], acc_rate: float | None = None):
        slug = slug.strip().lower()
        if not slug:
            return
        if slug in catalogue_map:
            # Update missing fields if available
            existing = catalogue_map[slug]
            if not existing["topic_tags"] and tags:
                existing["topic_tags"] = tags
            return

        clean_tags = []
        for t in tags:
            for canon in config.TOPIC_TAGS:
                if canon.lower() == t.lower() or (t.lower() in canon.lower() and len(t) > 3):
                    if canon not in clean_tags:
                        clean_tags.append(canon)
        if not clean_tags:
            clean_tags = _infer_topic_tags_from_slug_or_title(slug, title)
            clean_tags = [t for t in clean_tags if t in config.TOPIC_TAGS]
        if not clean_tags:
            clean_tags = ["Array"]

        if difficulty not in config.DIFFICULTIES:
            difficulty = "Medium"

        if acc_rate is None:
            base_acc = 0.70 if difficulty == "Easy" else 0.48 if difficulty == "Medium" else 0.32
            acc_rate = round(base_acc, 2)

        desc = _generate_problem_description(title, difficulty, clean_tags)
        catalogue_map[slug] = {
            "slug": slug,
            "title": title,
            "difficulty": difficulty,
            "topic_tags": clean_tags,
            "acceptance_rate": acc_rate,
            "description": desc,
        }

    # 1. Add well-known reference questions first
    for ref in REFERENCE_QUESTIONS:
        add_question(ref["slug"], ref["title"], ref["difficulty"], ref["topic_tags"], ref.get("acceptance_rate"))

    # 2. Ingest from sync_store.json and large_user.json
    sync_p = config.FILES_DIR / "sync_store.json"
    demo_p = config.FILES_DIR / "demo" / "large_user.json"

    all_subs = []
    if sync_p.exists():
        try:
            for acc in json.loads(sync_p.read_text(encoding="utf-8")).values():
                all_subs.extend(acc.get("submissions", []))
        except Exception:
            pass

    if demo_p.exists():
        try:
            all_subs.extend(json.loads(demo_p.read_text(encoding="utf-8")).get("submissions", []))
        except Exception:
            pass

    for s in all_subs:
        slug = str(s.get("problem_slug") or s.get("question_id") or "").strip().lower()
        if not slug or slug.isdigit():
            continue
        title = s.get("problem_title") or s.get("title") or slug.replace("-", " ").title()
        diff = s.get("difficulty", "Medium")
        raw_tags = s.get("topics") or []
        add_question(slug, title, diff, raw_tags)

    # 3. Ensure we have at least 450 canonical questions covering all topics
    # Generate canonical topic-focused questions if needed
    needed = max(0, config.NUM_QUESTIONS - len(catalogue_map))
    if needed > 0:
        idx = 1
        for topic in config.TOPIC_TAGS:
            for diff in config.DIFFICULTIES:
                slug = f"canonical-{topic.lower().replace(' ', '-')}-{diff.lower()}-{idx}"
                title = f"{topic} Mastery {diff} {idx}"
                add_question(slug, title, diff, [topic])
                idx += 1
                if len(catalogue_map) >= config.NUM_QUESTIONS:
                    break
            if len(catalogue_map) >= config.NUM_QUESTIONS:
                break

    # 4. Assign consistent 1-indexed question_id
    catalogue_list = []
    for qid, (slug, item) in enumerate(sorted(catalogue_map.items()), start=1):
        item["question_id"] = qid
        catalogue_list.append(item)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_OUTPUT.write_text(json.dumps(catalogue_list, indent=2), encoding="utf-8")
    print(f"Successfully generated {len(catalogue_list)} canonical questions at {CANONICAL_OUTPUT}")
    return catalogue_list


if __name__ == "__main__":
    build_canonical_catalogue()
