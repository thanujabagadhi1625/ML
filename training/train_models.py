"""
training/train_models.py
------------------------
OFFLINE TRAINING PIPELINE

Transforms the LeetCode Mentor system into a technically defensible offline-trained ML system:
  1. Canonical Question Catalogue: Loads or generates canonical question universe.
  2. Controlled Synthetic Population: Simulates realistic multi-user archetypes
     (strong overall, weak DP, weak graph, improving, declining, specialized, etc.).
  3. Strict User-Level Splitting: Partitions users into Train (70%), Val (15%), and Test (15%)
     with zero submission leakage across splits.
  4. Precomputed Embeddings: Generates and persists dense question embeddings (all-MiniLM-L6-v2).
  5. Feature Engineering: Computes observable feature matrices (dropping all latent variables).
  6. XGBoost Contest Rating Predictor: Fits on train users, validates on val users, evaluates
     on untouched test users, and serializes rating_model.pkl.
  7. ALS Implicit Matrix Factorization: Fits on train interactions over canonical questions,
     precomputes Y^T Y, and serializes als_model.npz.
  8. Recommendation Evaluation: Evaluates hybrid recommender with user fold-in on held-out
     temporal interactions from untouched test users (Precision@5, Recall@5, Hit Rate@5, NDCG@5).
  9. Weak-Topic Evaluation: Evaluates predictive power of historical topic risk profiling against
     actual future user failures on test users (Precision, Recall, F1).
  10. Model Metadata & Reproducibility: Serializes model_metadata.json with all metrics and parameters.

Run directly:
    python training/train_models.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root and files/ to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

import numpy as np
import pandas as pd
import sklearn
import xgboost
import torch

import config
from data_processing import (
    FeatureEngineer,
    compute_user_topic_profile,
    generate_synthetic_submissions,
    generate_synthetic_users,
    load_canonical_questions,
    temporal_split_user_submissions,
    user_train_val_test_split,
)
from nlp_cluster import WeakTopicAnalyzer
from predictor import ContestRatingPredictor
from recommender import (
    ALSMatrixFactorization,
    HybridRecommender,
    build_interaction_matrix,
)
from training.build_canonical_catalogue import build_canonical_catalogue


def evaluate_weak_topic_detection(
    test_user_ids: list[int],
    questions_df: pd.DataFrame,
    test_history_subs: pd.DataFrame,
    test_future_subs: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluates weak-topic detection as a predictive task:
      1. Uses historical interactions (first 80%) to compute topic risk and predict weak topics.
      2. Hides future submissions (last 20%).
      3. Compares predicted weak topics against topics where the user actually failed in the future.
      4. Reports Precision, Recall, and F1 across test users.
    """
    precisions = []
    recalls = []
    f1s = []

    # Identify topics where user actually failed in the future
    future_with_tags = test_future_subs.merge(
        questions_df[["question_id", "topic_tags"]], on="question_id", how="left"
    )
    future_with_tags = future_with_tags.explode("topic_tags").dropna(subset=["topic_tags"])

    future_failed_topics_by_user = {}
    for uid, grp in future_with_tags[future_with_tags["status"] != "Accepted"].groupby("user_id"):
        future_failed_topics_by_user[uid] = set(grp["topic_tags"].unique())

    for uid in test_user_ids:
        actual_failed_topics = future_failed_topics_by_user.get(uid, set())
        if not actual_failed_topics:
            continue

        u_hist = test_history_subs[test_history_subs["user_id"] == uid]
        if u_hist.empty:
            continue

        # Predict weak topics from history
        profile_df = compute_user_topic_profile(u_hist, questions_df, user_id=uid)
        if profile_df.empty:
            continue

        # Weak topics are those flagged as Critical or Weak
        predicted_weak = set(
            profile_df[profile_df["weakness_level"].isin(["Critical", "Weak"])]["topic"].unique()
        )
        if not predicted_weak:
            # Fallback to highest risk topics if none reached threshold
            predicted_weak = set(profile_df.head(2)["topic"].unique())

        tp = len(predicted_weak.intersection(actual_failed_topics))
        prec = tp / float(len(predicted_weak)) if predicted_weak else 0.0
        rec = tp / float(len(actual_failed_topics)) if actual_failed_topics else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "evaluated_users": len(precisions),
    }


def train_and_evaluate():
    print("=" * 70)
    print("LEETCODE MENTOR — OFFLINE TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Canonical Question Catalogue
    print("\n[1/7] Ensuring canonical question catalogue...")
    if not config.CANONICAL_QUESTIONS_PATH.exists():
        build_canonical_catalogue()
    questions_df = load_canonical_questions()
    print(f"  Loaded {len(questions_df)} canonical questions.")

    # 2. Precompute Question Embeddings
    print("\n[2/7] Precomputing question embeddings with sentence-transformers...")
    analyzer = WeakTopicAnalyzer()
    descriptions = questions_df["description"].fillna("").tolist()
    embeddings = analyzer.embed_descriptions(descriptions)
    config.QUESTION_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.QUESTION_EMBEDDINGS_PATH, embeddings)
    print(f"  Saved question embeddings to {config.QUESTION_EMBEDDINGS_PATH} (shape: {embeddings.shape})")

    # 3. Controlled Synthetic Multi-User Population Generation
    print("\n[3/7] Generating synthetic multi-user population with realistic archetypes...")
    users_df, latent_info = generate_synthetic_users(
        n=config.NUM_USERS,
        random_seed=config.RANDOM_SEED,
        return_latent_info=True,
    )
    submissions_df = generate_synthetic_submissions(
        users_df=users_df,
        questions_df=questions_df,
        n_submissions=config.NUM_SUBMISSIONS,
        random_seed=config.RANDOM_SEED,
        latent_users_info=latent_info,
    )
    print(f"  Generated {len(users_df)} synthetic users across {len(config.USER_ARCHETYPES)} archetypes.")
    print(f"  Generated {len(submissions_df)} simulated submission logs.")

    # 4. User-Level Train/Validation/Test Partitioning
    print("\n[4/7] Partitioning users into user-level splits (Train 70% | Val 15% | Test 15%)...")
    train_users, val_users, test_users, train_subs, val_subs, test_subs = user_train_val_test_split(
        users_df=users_df,
        submissions_df=submissions_df,
        train_ratio=config.TRAIN_USER_RATIO,
        val_ratio=config.VAL_USER_RATIO,
        test_ratio=config.TEST_USER_RATIO,
        random_seed=config.RANDOM_SEED,
    )
    print(f"  Train: {len(train_users)} users ({len(train_subs)} submissions)")
    print(f"  Val:   {len(val_users)} users ({len(val_subs)} submissions)")
    print(f"  Test:  {len(test_users)} users ({len(test_subs)} submissions)")

    # 5. Feature Engineering (Observable Features Only, Zero Latent Leakage)
    print("\n[5/7] Running vectorized feature engineering on user splits...")
    fe_train = FeatureEngineer(train_users, questions_df, train_subs)
    fe_val = FeatureEngineer(val_users, questions_df, val_subs)
    fe_test = FeatureEngineer(test_users, questions_df, test_subs)

    train_matrix = fe_train.build_user_feature_matrix()
    val_matrix = fe_val.build_user_feature_matrix()
    test_matrix = fe_test.build_user_feature_matrix()
    feature_cols = [c for c in train_matrix.columns if c not in ["user_id", "contest_rating"]]
    print(f"  Engineered {len(feature_cols)} observable features: {feature_cols}")

    # 6. Train & Evaluate XGBoost Rating Model
    print("\n[6/7] Training XGBoost contest rating model...")
    predictor = ContestRatingPredictor()
    val_metrics = predictor.fit(train_matrix=train_matrix, val_matrix=val_matrix, use_grid_search=False)
    test_metrics = predictor.evaluate(test_matrix=test_matrix)
    predictor.save_model(config.RATING_MODEL_PATH)
    print(f"  Validation Metrics: RMSE={val_metrics['rmse']:.2f}, MAE={val_metrics['mae']:.2f}, R2={val_metrics['r2']:.3f}")
    print(f"  Test Metrics (Untouched Users): RMSE={test_metrics['rmse']:.2f}, MAE={test_metrics['mae']:.2f}, R2={test_metrics['r2']:.3f}")
    print(f"  Saved rating model to {config.RATING_MODEL_PATH}")

    # 7. Train & Evaluate ALS Recommender + Hybrid Blending
    print("\n[7/7] Training ALS Recommender & Evaluating Recommendation / Weak-Topic Systems...")
    train_interactions = build_interaction_matrix(train_subs)
    als = ALSMatrixFactorization(
        n_factors=config.MF_LATENT_DIM,
        reg_lambda=config.MF_REG_LAMBDA,
        alpha=config.MF_CONFIDENCE_ALPHA,
        n_epochs=config.MF_EPOCHS,
        random_state=config.RANDOM_SEED,
    ).fit(train_interactions)
    als.save(config.ALS_MODEL_PATH)
    print(f"  Saved ALS model to {config.ALS_MODEL_PATH}")

    # Recommendation temporal evaluation on untouched test users
    test_hist_subs, test_fut_subs = temporal_split_user_submissions(
        test_subs, history_ratio=config.TEMPORAL_HISTORY_RATIO
    )
    test_user_ids = test_users["user_id"].tolist()

    recommender = HybridRecommender(questions_df, als_model=als)
    rec_eval = HybridRecommender.evaluate_recommendations(
        recommender=recommender,
        user_ids=test_user_ids,
        history_subs=test_hist_subs,
        future_subs=test_fut_subs,
        top_n=config.TOP_N_RECOMMENDATIONS,
    )
    print(f"  Recommendation Evaluation (Test Users, Temporal Held-Out Interactions):")
    print(f"    Precision@5: {rec_eval.get('precision@5', 0):.4f}")
    print(f"    Recall@5:    {rec_eval.get('recall@5', 0):.4f}")
    print(f"    Hit Rate@5:  {rec_eval.get('hit_rate@5', 0):.4f}")
    print(f"    NDCG@5:      {rec_eval.get('ndcg@5', 0):.4f}")

    # Weak topic prediction evaluation on untouched test users
    weak_topic_eval = evaluate_weak_topic_detection(
        test_user_ids=test_user_ids,
        questions_df=questions_df,
        test_history_subs=test_hist_subs,
        test_future_subs=test_fut_subs,
    )
    print(f"  Weak-Topic Detection Evaluation (Predicting Future Failures):")
    print(f"    Precision: {weak_topic_eval.get('precision', 0):.4f}")
    print(f"    Recall:    {weak_topic_eval.get('recall', 0):.4f}")
    print(f"    F1 Score:  {weak_topic_eval.get('f1', 0):.4f}")

    # 8. Model Metadata Persistence
    metadata = {
        "model_version": "2.0.0",
        "training_dataset_version": "synthetic_archetypes_v2",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.RANDOM_SEED,
        "n_synthetic_users": len(users_df),
        "n_questions": len(questions_df),
        "n_submissions": len(submissions_df),
        "user_split": {
            "train_users": len(train_users),
            "val_users": len(val_users),
            "test_users": len(test_users),
        },
        "feature_names": feature_cols,
        "xgboost_hyperparameters": config.XGB_PARAMS,
        "als_hyperparameters": {
            "latent_dim": config.MF_LATENT_DIM,
            "reg_lambda": config.MF_REG_LAMBDA,
            "confidence_alpha": config.MF_CONFIDENCE_ALPHA,
            "epochs": config.MF_EPOCHS,
        },
        "hybrid_blending_weights": {
            "cf_weight": config.HYBRID_CF_WEIGHT,
            "content_weight": config.HYBRID_CONTENT_WEIGHT,
            "weakness_weight": config.HYBRID_WEAKNESS_WEIGHT,
        },
        "evaluation_metrics": {
            "rating_model_test": {
                "rmse": round(test_metrics["rmse"], 2),
                "mae": round(test_metrics["mae"], 2),
                "r2": round(test_metrics["r2"], 3),
            },
            "recommendation_test": {
                "precision_at_5": round(rec_eval.get("precision@5", 0.0), 4),
                "recall_at_5": round(rec_eval.get("recall@5", 0.0), 4),
                "hit_rate_at_5": round(rec_eval.get("hit_rate@5", 0.0), 4),
                "ndcg_at_5": round(rec_eval.get("ndcg@5", 0.0), 4),
            },
            "weak_topic_detection_test": {
                "precision": round(weak_topic_eval.get("precision", 0.0), 4),
                "recall": round(weak_topic_eval.get("recall", 0.0), 4),
                "f1": round(weak_topic_eval.get("f1", 0.0), 4),
            },
        },
        "environment": {
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "disclaimer": (
            "This model was trained on a controlled synthetic multi-user population. "
            "Real-world generalization to live LeetCode contest ratings is benchmark evidence "
            "and has not been verified on real labeled multi-user contest datasets."
        ),
    }

    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nModel metadata successfully serialized to {config.MODEL_METADATA_PATH}")
    print("\n" + "=" * 70)
    print("OFFLINE TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
