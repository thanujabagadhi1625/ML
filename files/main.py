"""
main.py
-------
End-to-end orchestration: wires DATA ENGINE -> WEAK TOPICS ENGINE ->
RECOMMENDATION ENGINE -> PREDICTION ENGINE into a single "mentor report"
for a given user.

Run directly:  python main.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import config
from data_processing import (
    FeatureEngineer,
    generate_full_synthetic_dataset,
    load_leetcode_history_export,
)
from nlp_cluster import WeakTopicAnalyzer
from predictor import ContestRatingPredictor
from recommender import HybridRecommender


class LeetCodeMentor:
    """Facade class -- loads pre-trained models and serves per-user reports."""

    def __init__(self, users_df, questions_df, submissions_df):
        self.users_df = users_df
        self.questions_df = questions_df
        self.submissions_df = submissions_df

        print("[1/4] Running feature engineering pipeline ...")
        self.feature_engineer = FeatureEngineer(self.users_df, self.questions_df, self.submissions_df)
        self.feature_matrix = self.feature_engineer.build_user_feature_matrix()

        print("[2/4] Initializing recommendation engine (ALS fold-in + content-based) ...")
        self.recommender = HybridRecommender(self.questions_df, self.submissions_df)

        print("[3/4] Loading offline-trained contest rating predictor (XGBoost) ...")
        self.predictor = ContestRatingPredictor()
        if config.RATING_MODEL_PATH.exists():
            try:
                self.predictor.load_model(config.RATING_MODEL_PATH)
            except Exception as e:
                print(f"Warning: Failed to load rating model from {config.RATING_MODEL_PATH}: {e}")

        # Load offline evaluation metrics from model_metadata.json if available
        self.training_metrics = {}
        if config.MODEL_METADATA_PATH.exists():
            try:
                meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
                test_eval = meta.get("evaluation_metrics", {}).get("rating_model_test", {})
                self.training_metrics = {
                    "available": True,
                    "rmse": test_eval.get("rmse"),
                    "mae": test_eval.get("mae"),
                    "r2": test_eval.get("r2"),
                    "metadata": meta,
                }
            except Exception:
                self.training_metrics = {"available": False, "note": "Failed to read model_metadata.json"}
        else:
            self.training_metrics = {
                "available": False,
                "note": "Model metadata not found. Run training/train_models.py to train offline models.",
            }

        print("[4/4] Initializing weak-topic analyzer (SentenceTransformer + c-TF-IDF) ...")
        self.topic_analyzer = WeakTopicAnalyzer()

        print("\nAll engines ready.\n")

    @staticmethod
    def print_report_summary(report: Dict):
        weak_clusters = report.get("weak_subtopics", {}).get("clusters", [])
        recommendations = report.get("recommended_questions", [])

        print("\n=== WEAK TOPICS ===")
        if not weak_clusters:
            print("No weak-topic clusters available yet.")
        else:
            for cluster in weak_clusters:
                print(
                    f"Cluster {cluster.get('cluster_id')}: "
                    f"size={cluster.get('size')} | "
                    f"keywords={cluster.get('keywords')}"
                )

        print("\n=== RECOMMENDED QUESTIONS ===")
        if not recommendations:
            print("No recommendations generated.")
        else:
            for i, item in enumerate(recommendations, start=1):
                lid = item.get("leetcode_id", item.get("question_id"))
                slug = item.get("slug", "")
                title = item.get("title", "")
                url = f"https://leetcode.com/problems/{slug}/" if slug else ""
                url_str = f" ({url})" if url else ""
                print(
                    f"{i}. #{lid} {title}{url_str} | "
                    f"{item.get('difficulty')} | tags={item.get('topic_tags')} | "
                    f"score={item.get('recommendation_score', 0):.4f}"
                )

        print("\n=== MODEL METRICS ===")
        metrics = report.get("model_evaluation", {})
        print(f"RMSE={metrics.get('rmse')} | MAE={metrics.get('mae')} | R2={metrics.get('r2')}")

    def generate_report(self, user_id: int, provenance: Dict | None = None, **kwargs) -> Dict:
        from data_processing import calculate_data_diagnostics

        diagnostics = calculate_data_diagnostics(self.submissions_df, self.questions_df)

        # 1. Structured Topic Weakness Profile
        topic_profile_df = self.feature_engineer.get_topic_profile(user_id=user_id)
        topic_profile_records = topic_profile_df.to_dict(orient="records") if not topic_profile_df.empty else []

        # 2. Secondary NLP Weak Sub-topics
        failed = self.feature_engineer.get_failed_submissions_with_text(user_id=user_id)
        weak_topics = self.topic_analyzer.analyze_user_weak_topics(failed)

        # 3. Recommended questions (profile-guided & evidence-grounded via ALS fold-in + content)
        user_subs = self.submissions_df[self.submissions_df["user_id"] == user_id]
        recommendations = self.recommender.recommend(
            user_id=user_id,
            topic_profile=topic_profile_df,
            user_submissions_df=user_subs,
            top_k=kwargs.get("top_k", 5),
        ).to_dict(orient="records")

        # 4. Predicted contest rating
        user_features = self.feature_matrix[self.feature_matrix["user_id"] == user_id]
        if self.predictor.is_fitted and not user_features.empty:
            predicted_rating = float(round(self.predictor.predict(user_features)[0], 1))
        else:
            predicted_rating = None

        eval_dict = {}
        if self.training_metrics.get("available"):
            eval_dict = {
                "available": True,
                "rmse": round(self.training_metrics["rmse"], 2) if self.training_metrics.get("rmse") is not None else None,
                "mae": round(self.training_metrics["mae"], 2) if self.training_metrics.get("mae") is not None else None,
                "r2": round(self.training_metrics["r2"], 3) if self.training_metrics.get("r2") is not None else None,
                "benchmark_note": "Trained offline on synthetic benchmark population; serves as proxy estimate.",
            }
        else:
            eval_dict = {
                "available": False,
                "note": self.training_metrics.get("note", "Rating prediction model not loaded. Run python training/train_models.py first."),
            }

        report = {
            "user_id": user_id,
            "data_provenance": provenance or {
                "data_source": "local_pipeline",
                "is_demo_data": False,
                "submission_count": len(self.submissions_df),
            },
            "diagnostics": diagnostics,
            "topic_profile": topic_profile_records,
            "weak_subtopics": weak_topics,
            "recommended_questions": recommendations,
            "predicted_contest_rating": round(predicted_rating, 1) if predicted_rating is not None else None,
            "model_evaluation": eval_dict,
        }

        try:
            report_file = Path(__file__).parent / "mentor_report.json"
            report_file.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LeetCode Mentor backend.")
    parser.add_argument(
        "--export",
        type=Path,
        help="Optional path to an exported LeetCode history file (JSON/CSV/TSV).",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        help="Optional user ID from the loaded dataset to generate a report for.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.export:
        print("[0/4] Loading exported LeetCode history ...")
        users_df, questions_df, submissions_df = load_leetcode_history_export(args.export)
    else:
        print("[0/4] Generating synthetic dataset ...")
        users_df, questions_df, submissions_df = generate_full_synthetic_dataset()

    mentor = LeetCodeMentor(users_df, questions_df, submissions_df)
    sample_user_id = args.user_id or int(mentor.users_df["user_id"].iloc[0])

    report = mentor.generate_report(sample_user_id)
    print(json.dumps(report, indent=2, default=str))
    mentor.print_report_summary(report)
