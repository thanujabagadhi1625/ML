"""
main.py
-------
End-to-end orchestration: wires DATA ENGINE -> WEAK TOPICS ENGINE ->
RECOMMENDATION ENGINE -> PREDICTION ENGINE into a single "mentor report"
for a given user.

Run directly:  python main.py
"""

from __future__ import annotations

import json
from typing import Dict

from data_processing import FeatureEngineer, generate_full_synthetic_dataset
from nlp_cluster import WeakTopicAnalyzer
from predictor import ContestRatingPredictor
from recommender import HybridRecommender


class LeetCodeMentor:
    """Facade class -- builds every engine once, then serves per-user reports."""

    def __init__(self):
        print("[1/4] Generating synthetic dataset ...")
        self.users_df, self.questions_df, self.submissions_df = generate_full_synthetic_dataset()

        print("[2/4] Running feature engineering pipeline ...")
        self.feature_engineer = FeatureEngineer(self.users_df, self.questions_df, self.submissions_df)
        self.feature_matrix = self.feature_engineer.build_user_feature_matrix()

        print("[3/4] Training recommendation engine (ALS + content-based) ...")
        self.recommender = HybridRecommender(self.questions_df, self.submissions_df)

        print("[4/4] Training contest rating predictor (XGBoost) ...")
        self.predictor = ContestRatingPredictor()
        self.training_metrics = self.predictor.fit(self.feature_matrix)

        print("Loading sentence-transformer for weak-topic analysis (lazy) ...")
        self.topic_analyzer = WeakTopicAnalyzer()

        print("\nAll engines ready.\n")

    def generate_report(self, user_id: int) -> Dict:
        # 1. Weak sub-topics
        failed = self.feature_engineer.get_failed_submissions_with_text(user_id=user_id)
        weak_topics = self.topic_analyzer.analyze_user_weak_topics(failed)

        # 2. Recommended questions
        recommendations = self.recommender.recommend(user_id).to_dict(orient="records")

        # 3. Predicted contest rating
        user_features = self.feature_matrix[self.feature_matrix["user_id"] == user_id]
        predicted_rating = float(self.predictor.predict(user_features)[0])

        return {
            "user_id": user_id,
            "weak_subtopics": weak_topics,
            "recommended_questions": recommendations,
            "predicted_contest_rating": round(predicted_rating, 1),
            "model_evaluation": {
                "rmse": round(self.training_metrics["rmse"], 2),
                "mae": round(self.training_metrics["mae"], 2),
                "r2": round(self.training_metrics["r2"], 3),
            },
        }


if __name__ == "__main__":
    mentor = LeetCodeMentor()
    sample_user_id = int(mentor.users_df["user_id"].iloc[0])

    report = mentor.generate_report(sample_user_id)
    print(json.dumps(report, indent=2, default=str))
