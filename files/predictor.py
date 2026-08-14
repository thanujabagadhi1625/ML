"""
predictor.py
------------
PREDICTION ENGINE

Supervised regression pipeline that predicts a user's contest rating from
their engineered feature vector (accuracy ratios, difficulty distribution,
recency/momentum -- everything produced by data_processing.FeatureEngineer).

Design choices:
  - sklearn Pipeline + ColumnTransformer so preprocessing is bundled with the
    model artifact (no train/serve skew).
  - XGBoost Regressor as the estimator, with a GridSearchCV hook for
    hyperparameter search (placeholders in config.XGB_PARAM_GRID).
  - RMSE / MAE / R^2 reported on a held-out test split.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config

TARGET_COLUMN = "contest_rating"
ID_COLUMNS = ["user_id"]


class ContestRatingPredictor:
    """Wraps the full preprocessing + XGBoost regression pipeline."""

    def __init__(self, xgb_params: Dict | None = None):
        self.xgb_params = xgb_params or config.XGB_PARAMS
        self.pipeline: Pipeline | None = None
        self.feature_columns_: list[str] | None = None
        self.is_trained: bool = False

    # ---- Pipeline construction --------------------------------------------------
    def _build_pipeline(self, numeric_features: list[str]) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_features),
            ],
            remainder="drop",
        )
        model = XGBRegressor(**self.xgb_params)
        return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # ---- Train/test split --------------------------------------------------------
    @staticmethod
    def split_features_target(feature_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        drop_cols = [c for c in ID_COLUMNS + [TARGET_COLUMN] if c in feature_matrix.columns]
        X = feature_matrix.drop(columns=drop_cols)
        y = feature_matrix[TARGET_COLUMN] if TARGET_COLUMN in feature_matrix.columns else pd.Series([1500.0] * len(feature_matrix))
        return X, y

    def fit(self, feature_matrix: pd.DataFrame, use_grid_search: bool = False) -> Dict:
        """
        Trains the pipeline on `feature_matrix` (output of
        FeatureEngineer.build_user_feature_matrix()), evaluates on a held-out
        split, and returns evaluation metrics.
        """
        X, y = self.split_features_target(feature_matrix)
        self.feature_columns_ = list(X.columns)

        if len(X) < 5:
            self.is_trained = False
            return {
                "available": False,
                "note": "Rating prediction unavailable due to insufficient training information (requires multiple independent user profiles).",
                "rmse": None,
                "mae": None,
                "r2": None,
                "best_params": self.xgb_params,
            }

        self.pipeline = self._build_pipeline(self.feature_columns_)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
        )

        if use_grid_search:
            search = GridSearchCV(
                self.pipeline,
                param_grid=config.XGB_PARAM_GRID,
                scoring="neg_root_mean_squared_error",
                cv=config.CV_FOLDS,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            self.pipeline = search.best_estimator_
            best_params = search.best_params_
        else:
            self.pipeline.fit(X_train, y_train)
            best_params = self.xgb_params

        self.is_trained = True
        y_pred = self.pipeline.predict(X_test)
        metrics = self._evaluate(y_test, y_pred)
        metrics["available"] = True
        metrics["best_params"] = best_params
        return metrics

    # ---- Evaluation --------------------------------------------------------------
    @staticmethod
    def _evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    # ---- Inference -----------------------------------------------------------------
    def predict(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """Predicts contest rating for new users or computes heuristic rating if model is untrained."""
        if self.is_trained and self.pipeline is not None:
            X = feature_matrix[self.feature_columns_]
            return self.pipeline.predict(X)
        
        # Heuristic fallback rating based on user accuracy & difficulty profile
        ratings = []
        for _, row in feature_matrix.iterrows():
            overall_acc = row.get("overall_accuracy", 0.5)
            hard_acc = row.get("accuracy_hard", 0.0)
            med_acc = row.get("accuracy_medium", 0.0)
            share_hard = row.get("share_hard", 0.0)

            est = 1200.0 + 400.0 * overall_acc + 300.0 * hard_acc + 150.0 * med_acc + 200.0 * share_hard
            ratings.append(float(np.clip(est, 800.0, 3000.0)))
        return np.array(ratings)

    # ---- Feature importance ---------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Returns feature importances from the trained XGBoost model, sorted descending."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet -- call fit() first.")
        model: XGBRegressor = self.pipeline.named_steps["model"]
        importances = model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_columns_, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    from data_processing import generate_full_synthetic_dataset, FeatureEngineer

    users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
    fe = FeatureEngineer(users_df, questions_df, submissions_df)
    feature_matrix = fe.build_user_feature_matrix()

    predictor = ContestRatingPredictor()
    metrics = predictor.fit(feature_matrix, use_grid_search=False)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nTop feature importances:")
    print(predictor.feature_importance().head(10).to_string(index=False))
