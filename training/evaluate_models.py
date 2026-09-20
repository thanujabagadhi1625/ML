"""
training/evaluate_models.py
---------------------------
Loads serialized model artifacts from models/ and evaluates performance on held-out test users:
  1. XGBoost Rating Predictor: RMSE, MAE, R^2
  2. Hybrid Recommender: Precision@5, Recall@5, Hit Rate@5, NDCG@5
  3. Weak Topic Detection: Precision, Recall, F1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
    generate_synthetic_dataset_v3,
    generate_synthetic_submissions,
    generate_synthetic_users,
    load_canonical_questions,
    temporal_split_user_submissions,
    user_train_val_test_split,
)
from predictor import ContestRatingPredictor
from recommender import ALSMatrixFactorization, HybridRecommender
from training.train_models import evaluate_weak_topic_detection


def run_evaluation():
    print("=" * 70)
    print("LEETCODE MENTOR — ARTIFACT EVALUATION")
    print("=" * 70)

    if not config.MODEL_METADATA_PATH.exists():
        print(f"Error: Model metadata not found at '{config.MODEL_METADATA_PATH}'.")
        print("Please run offline training first: python training/train_models.py")
        return

    metadata = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    print(f"\nModel Version: {metadata.get('model_version')}")
    print(f"Trained At:    {metadata.get('training_timestamp')}")
    print(f"Random Seed:   {metadata.get('random_seed')}")

    print("\nExisting Offline Evaluation Metrics:")
    print(json.dumps(metadata.get("evaluation_metrics", {}), indent=2))

    # Re-evaluate live against a fresh test set for verification
    print("\nRe-evaluating artifacts against held-out benchmark population...")
    questions_df = load_canonical_questions()
    if config.CURRENT_PROFILE == "v3":
        users_df, questions_df, subs_df, _ = generate_synthetic_dataset_v3(
            questions_df=questions_df,
            n_users=config.NUM_USERS,
            random_seed=config.RANDOM_SEED,
            zipf_exponent=0.8,
            beta=1.5,
        )
    else:
        users_df, latent_info = generate_synthetic_users(n=config.NUM_USERS, return_latent_info=True)
        subs_df = generate_synthetic_submissions(users_df, questions_df, latent_users_info=latent_info)

    _, _, test_users, _, _, test_subs = user_train_val_test_split(
        users_df, subs_df, random_seed=config.RANDOM_SEED
    )

    fe_test = FeatureEngineer(test_users, questions_df, test_subs)
    test_matrix = fe_test.build_user_feature_matrix()

    # 1. Rating Predictor
    predictor = ContestRatingPredictor()
    predictor.load_model(config.RATING_MODEL_PATH)
    rating_metrics = predictor.evaluate(test_matrix)
    print("\n1. Rating Regression Evaluation (Live Re-check on Test Users):")
    print(f"   RMSE: {rating_metrics['rmse']:.2f} | MAE: {rating_metrics['mae']:.2f} | R^2: {rating_metrics['r2']:.3f}")

    # 2. Recommendation Evaluation
    als = ALSMatrixFactorization().load(config.ALS_MODEL_PATH)
    recommender = HybridRecommender(questions_df, als_model=als)
    test_hist, test_fut = temporal_split_user_submissions(test_subs)
    test_uids = test_users["user_id"].tolist()

    rec_metrics = HybridRecommender.evaluate_recommendations(
        recommender=recommender,
        user_ids=test_uids,
        history_subs=test_hist,
        future_subs=test_fut,
        top_n=config.TOP_N_RECOMMENDATIONS,
    )
    print("\n2. Recommendation Evaluation (Live Re-check on Test Users):")
    print(f"   Precision@5: {rec_metrics.get('precision@5', 0):.4f}")
    print(f"   Recall@5:    {rec_metrics.get('recall@5', 0):.4f}")
    print(f"   Hit Rate@5:  {rec_metrics.get('hit_rate@5', 0):.4f}")
    print(f"   NDCG@5:      {rec_metrics.get('ndcg@5', 0):.4f}")

    # 3. Weak-Topic Detection
    weak_metrics = evaluate_weak_topic_detection(
        test_user_ids=test_uids,
        questions_df=questions_df,
        test_history_subs=test_hist,
        test_future_subs=test_fut,
    )
    print("\n3. Weak-Topic Evaluation (Live Re-check on Test Users):")
    print(f"   Precision: {weak_metrics.get('precision', 0):.4f}")
    print(f"   Recall:    {weak_metrics.get('recall', 0):.4f}")
    print(f"   F1 Score:  {weak_metrics.get('f1', 0):.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_evaluation()
