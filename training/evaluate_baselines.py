"""
training/evaluate_baselines.py
------------------------------
Evaluates offline baselines across 10 random seeds on the canonical catalogue (v3 generator):
1. Contest Rating Model Baselines:
   - Predict-the-Mean (train target mean)
   - Linear Regression (StandardScaler + LinearRegression)
   - XGBoost Regressor (production architecture)
   Metrics: R^2, RMSE, MAE (mean and std across 10 seeds)

2. Weak-Topic Detection Baselines:
   - All attempted topics in history
   - Top-2 topics by lifetime failure rate in history
   - Random 2 topics in history
   - System (composite risk score with Critical/Weak classification)
   Metrics: Precision, Recall, F1 (mean and std across 10 seeds)

Saves results to:
  - results/rating_model_baselines.json
  - results/weak_topic_baselines.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root and files/ to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config
from data_processing import (
    FeatureEngineer,
    compute_user_topic_profile,
    generate_synthetic_dataset_v3,
    load_canonical_questions,
    temporal_split_user_submissions,
    user_train_val_test_split,
)


def evaluate_rating_baselines_on_seed(
    train_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    target = "contest_rating"
    id_col = "user_id"
    feature_cols = [c for c in train_matrix.columns if c not in [id_col, target]]

    X_train = train_matrix[feature_cols].values
    y_train = train_matrix[target].values
    X_test = test_matrix[feature_cols].values
    y_test = test_matrix[target].values

    # 1. Predict-the-mean
    mean_val = float(np.mean(y_train))
    y_pred_mean = np.full_like(y_test, mean_val)
    r2_mean = float(r2_score(y_test, y_pred_mean))
    rmse_mean = float(np.sqrt(mean_squared_error(y_test, y_pred_mean)))
    mae_mean = float(mean_absolute_error(y_test, y_pred_mean))

    # 2. Linear Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    r2_lr = float(r2_score(y_test, y_pred_lr))
    rmse_lr = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    mae_lr = float(mean_absolute_error(y_test, y_pred_lr))

    # 3. XGBoost
    xgb_params = dict(config.XGB_PARAMS)
    xgb_params["random_state"] = seed
    xgb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", XGBRegressor(**xgb_params)),
    ])
    xgb_pipe.fit(X_train, y_train)
    y_pred_xgb = xgb_pipe.predict(X_test)
    r2_xgb = float(r2_score(y_test, y_pred_xgb))
    rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
    mae_xgb = float(mean_absolute_error(y_test, y_pred_xgb))

    return {
        "predict_the_mean": {"r2": r2_mean, "rmse": rmse_mean, "mae": mae_mean},
        "linear_regression": {"r2": r2_lr, "rmse": rmse_lr, "mae": mae_lr},
        "xgboost": {"r2": r2_xgb, "rmse": rmse_xgb, "mae": mae_xgb},
    }


def evaluate_weak_topic_baselines_on_seed(
    test_user_ids: List[int],
    questions_df: pd.DataFrame,
    test_history_subs: pd.DataFrame,
    test_future_subs: pd.DataFrame,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    # Merge future submissions with tags
    fut = test_future_subs.merge(
        questions_df[["question_id", "topic_tags"]], on="question_id", how="left"
    )
    fut = fut.explode("topic_tags").dropna(subset=["topic_tags"])
    fut["topic_tags"] = fut["topic_tags"].astype(str).str.strip()
    fut = fut[fut["topic_tags"] != ""]

    future_failed_by_user = {}
    for uid, grp in fut[fut["status"] != "Accepted"].groupby("user_id"):
        future_failed_by_user[uid] = set(grp["topic_tags"].unique())

    # Pre-merge history submissions with tags
    hist = test_history_subs.merge(
        questions_df[["question_id", "topic_tags"]], on="question_id", how="left"
    )
    hist = hist.explode("topic_tags").dropna(subset=["topic_tags"])
    hist["topic_tags"] = hist["topic_tags"].astype(str).str.strip()
    hist = hist[hist["topic_tags"] != ""]

    methods = ["system", "all_attempted", "top2_failure_rate", "random_2"]
    user_scores = {m: {"prec": [], "rec": [], "f1": []} for m in methods}

    for uid in test_user_ids:
        actual_failed = future_failed_by_user.get(uid, set())
        if not actual_failed:
            continue

        u_hist = test_history_subs[test_history_subs["user_id"] == uid]
        if u_hist.empty:
            continue

        u_hist_tags = hist[hist["user_id"] == uid]
        if u_hist_tags.empty:
            continue

        attempted_topics = list(u_hist_tags["topic_tags"].unique())
        if not attempted_topics:
            continue

        # 1. System (Critical/Weak composite risk)
        profile_df = compute_user_topic_profile(u_hist, questions_df, user_id=uid)
        if not profile_df.empty:
            pred_sys = set(
                profile_df[profile_df["weakness_level"].isin(["Critical", "Weak"])]["topic"].unique()
            )
            if not pred_sys:
                pred_sys = set(profile_df.head(2)["topic"].unique())
        else:
            pred_sys = set(attempted_topics[:2])

        # 2. All attempted topics
        pred_all = set(attempted_topics)

        # 3. Top-2 topics by lifetime failure rate
        topic_stats = []
        for t_name, t_grp in u_hist_tags.groupby("topic_tags"):
            att = len(t_grp)
            fails = int((t_grp["status"] != "Accepted").sum())
            fail_rate = fails / att if att > 0 else 0.0
            topic_stats.append((t_name, fail_rate, fails, att))
        # Sort descending by fail_rate, then fails, then att
        topic_stats.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        pred_top2 = set([t[0] for t in topic_stats[:2]])

        # 4. Random 2 topics
        rng = np.random.RandomState(seed * 10007 + uid)
        n_pick = min(2, len(attempted_topics))
        pred_rnd = set(rng.choice(attempted_topics, size=n_pick, replace=False))

        preds_dict = {
            "system": pred_sys,
            "all_attempted": pred_all,
            "top2_failure_rate": pred_top2,
            "random_2": pred_rnd,
        }

        for m in methods:
            p_set = preds_dict[m]
            tp = len(p_set.intersection(actual_failed))
            prec = tp / float(len(p_set)) if p_set else 0.0
            rec = tp / float(len(actual_failed)) if actual_failed else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            user_scores[m]["prec"].append(prec)
            user_scores[m]["rec"].append(rec)
            user_scores[m]["f1"].append(f1)

    seed_result = {}
    for m in methods:
        seed_result[m] = {
            "precision": float(np.mean(user_scores[m]["prec"])) if user_scores[m]["prec"] else 0.0,
            "recall": float(np.mean(user_scores[m]["rec"])) if user_scores[m]["rec"] else 0.0,
            "f1": float(np.mean(user_scores[m]["f1"])) if user_scores[m]["f1"] else 0.0,
            "evaluated_users": len(user_scores[m]["f1"]),
        }
    return seed_result


def main():
    print("=" * 70)
    print("EVALUATING MODEL BASELINES ACROSS 10 SEEDS")
    print("=" * 70)

    canonical_df = load_canonical_questions()
    print(f"Loaded canonical questions: {len(canonical_df)}")

    seeds = list(range(1, 11))
    n_users = 2000

    rating_records = {
        "predict_the_mean": {"r2": [], "rmse": [], "mae": []},
        "linear_regression": {"r2": [], "rmse": [], "mae": []},
        "xgboost": {"r2": [], "rmse": [], "mae": []},
    }

    weak_topic_records = {
        "system": {"precision": [], "recall": [], "f1": []},
        "all_attempted": {"precision": [], "recall": [], "f1": []},
        "top2_failure_rate": {"precision": [], "recall": [], "f1": []},
        "random_2": {"precision": [], "recall": [], "f1": []},
    }

    for seed in seeds:
        print(f"\n--- Running Seed {seed}/10 ---")
        users_df, questions_df, subs_df, _ = generate_synthetic_dataset_v3(
            canonical_df,
            n_users=n_users,
            random_seed=seed,
            zipf_exponent=0.8,
            beta=1.5,
        )

        train_u, val_u, test_u, train_s, val_s, test_s = user_train_val_test_split(
            users_df=users_df,
            submissions_df=subs_df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=seed,
        )

        # 1. Feature Matrices for Rating Predictor
        fe_train = FeatureEngineer(train_u, questions_df, train_s)
        fe_test = FeatureEngineer(test_u, questions_df, test_s)
        train_mat = fe_train.build_user_feature_matrix()
        test_mat = fe_test.build_user_feature_matrix()

        r_res = evaluate_rating_baselines_on_seed(train_mat, test_mat, seed=seed)
        for m in rating_records:
            for metric in ["r2", "rmse", "mae"]:
                rating_records[m][metric].append(r_res[m][metric])

        print(
            f"  Rating R^2 -> Mean: {r_res['predict_the_mean']['r2']:.4f} | "
            f"LinReg: {r_res['linear_regression']['r2']:.4f} | "
            f"XGB: {r_res['xgboost']['r2']:.4f}"
        )

        # 2. Weak Topic Baselines
        test_hist, test_fut = temporal_split_user_submissions(test_s, history_ratio=0.80)
        test_uids = test_u["user_id"].tolist()
        wt_res = evaluate_weak_topic_baselines_on_seed(
            test_user_ids=test_uids,
            questions_df=questions_df,
            test_history_subs=test_hist,
            test_future_subs=test_fut,
            seed=seed,
        )
        for m in weak_topic_records:
            for metric in ["precision", "recall", "f1"]:
                weak_topic_records[m][metric].append(wt_res[m][metric])

        print(
            f"  Weak-Topic F1 -> System: {wt_res['system']['f1']:.4f} | "
            f"All: {wt_res['all_attempted']['f1']:.4f} | "
            f"Top2: {wt_res['top2_failure_rate']['f1']:.4f} | "
            f"Rnd2: {wt_res['random_2']['f1']:.4f}"
        )

    # Compute summary stats
    rating_summary = {
        "seeds": seeds,
        "n_users": n_users,
        "catalogue_size": len(canonical_df),
        "generator_version": "v3",
        "models": {},
    }
    for m, metrics in rating_records.items():
        rating_summary["models"][m] = {}
        for metric, vals in metrics.items():
            rating_summary["models"][m][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_seed": [float(v) for v in vals],
            }

    weak_topic_summary = {
        "seeds": seeds,
        "n_users": n_users,
        "catalogue_size": len(canonical_df),
        "generator_version": "v3",
        "methods": {},
    }
    for m, metrics in weak_topic_records.items():
        weak_topic_summary["methods"][m] = {}
        for metric, vals in metrics.items():
            weak_topic_summary["methods"][m][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_seed": [float(v) for v in vals],
            }

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    r_path = results_dir / "rating_model_baselines.json"
    r_path.write_text(json.dumps(rating_summary, indent=2), encoding="utf-8")
    print(f"\nSaved rating model baselines to {r_path}")

    wt_path = results_dir / "weak_topic_baselines.json"
    wt_path.write_text(json.dumps(weak_topic_summary, indent=2), encoding="utf-8")
    print(f"Saved weak-topic baselines to {wt_path}")


if __name__ == "__main__":
    main()
