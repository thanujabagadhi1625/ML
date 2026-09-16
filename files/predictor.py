"""
predictor.py
------------
PREDICTION ENGINE

Supervised regression pipeline that predicts a user's contest rating from
their engineered feature vector (accuracy ratios, difficulty distribution,
recency momentum, submission intensity -- produced by FeatureEngineer).

Architecture:
  - OFFLINE: Trained strictly on synthetic multi-user population with user-level
    separation (train/val/test). Evaluated on untouched test users and saved to models/rating_model.pkl.
  - ONLINE: Loads pre-trained model artifact and performs inference on a single
    real user's engineered feature vector.
  - No retraining happens during normal real-user inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
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
    """Wraps the preprocessing + XGBoost regression pipeline with artifact save/load support."""

    def __init__(self, xgb_params: Dict | None = None):
        self.xgb_params = xgb_params or config.XGB_PARAMS
        self.pipeline: Pipeline | None = None
        self.feature_columns_: list[str] | None = None
        self.is_trained: bool = False

    @property
    def is_fitted(self) -> bool:
        return self.is_trained

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

    # ---- Offline Training --------------------------------------------------------
    def fit(
        self,
        train_matrix: pd.DataFrame,
        val_matrix: pd.DataFrame | None = None,
        use_grid_search: bool = False,
    ) -> Dict:
        """
        Trains the pipeline on user feature matrix (offline multi-user training).
        Evaluates on held-out validation users and returns evaluation metrics.
        """
        X_train, y_train = self.split_features_target(train_matrix)
        self.feature_columns_ = list(X_train.columns)

        if len(X_train) < 10:
            self.is_trained = False
            return {
                "available": False,
                "note": "Rating prediction model requires multi-user distribution for offline training.",
                "rmse": None,
                "mae": None,
                "r2": None,
                "best_params": self.xgb_params,
            }

        self.pipeline = self._build_pipeline(self.feature_columns_)

        if val_matrix is not None and not val_matrix.empty:
            X_eval, y_eval = self.split_features_target(val_matrix)
        else:
            X_train, X_eval, y_train, y_eval = train_test_split(
                X_train, y_train, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
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
            preprocessor = self.pipeline.named_steps["preprocessor"]
            X_tr_proc = preprocessor.fit_transform(X_train)
            X_ev_proc = preprocessor.transform(X_eval)

            model = self.pipeline.named_steps["model"]
            if len(X_eval) >= 5:
                model.set_params(early_stopping_rounds=25)
                model.fit(X_tr_proc, y_train, eval_set=[(X_ev_proc, y_eval)], verbose=False)
            else:
                model.fit(X_tr_proc, y_train, verbose=False)
            best_params = self.xgb_params

        self.is_trained = True
        y_pred = self.pipeline.predict(X_eval)
        metrics = self._evaluate(y_eval, y_pred)
        metrics["available"] = True
        metrics["best_params"] = best_params
        return metrics

    # ---- Evaluation --------------------------------------------------------------
    @staticmethod
    def _evaluate(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def evaluate(self, test_matrix: pd.DataFrame) -> Dict:
        """Evaluates trained pipeline on held-out untouched test users."""
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model has not been trained yet -- call fit() or load_model() first.")
        X_test, y_test = self.split_features_target(test_matrix)
        X = pd.DataFrame(index=X_test.index)
        for col in self.feature_columns_:
            X[col] = X_test[col] if col in X_test.columns else 0.0
        y_pred = self.pipeline.predict(X)
        metrics = self._evaluate(y_test, y_pred)
        metrics["available"] = True
        return metrics

    # ---- Artifact Persistence ---------------------------------------------------
    def save_model(self, model_path: str | Path = config.RATING_MODEL_PATH) -> Path:
        """Serializes the trained pipeline and feature column list to disk."""
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Cannot save an untrained model pipeline.")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "feature_columns": self.feature_columns_,
            "xgb_params": self.xgb_params,
        }
        joblib.dump(payload, path)
        return path

    def load_model(self, model_path: str | Path = config.RATING_MODEL_PATH) -> ContestRatingPredictor:
        """Loads a pre-trained pipeline and feature list from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Rating model artifact not found at '{path}'. "
                "Please run offline training first: python training/train_models.py"
            )
        payload = joblib.load(path)
        self.pipeline = payload["pipeline"]
        self.feature_columns_ = payload["feature_columns"]
        self.xgb_params = payload.get("xgb_params", self.xgb_params)
        self.is_trained = True
        return self

    # ---- Inference ---------------------------------------------------------------
    def predict(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """
        Predicts contest rating for user(s) using the loaded offline model.
        Fills missing feature columns with 0.0 to ensure schema compatibility.
        Does NOT retrain during inference.
        """
        if not self.is_trained or self.pipeline is None:
            if config.RATING_MODEL_PATH.exists():
                self.load_model(config.RATING_MODEL_PATH)
            else:
                raise RuntimeError(
                    f"ContestRatingPredictor is not trained and no artifact exists at '{config.RATING_MODEL_PATH}'. "
                    "Run offline training first: python training/train_models.py"
                )

        X = pd.DataFrame(index=feature_matrix.index)
        for col in self.feature_columns_:
            X[col] = feature_matrix[col] if col in feature_matrix.columns else 0.0

        preds = self.pipeline.predict(X)
        return np.clip(preds, 800.0, 3000.0)

    # ---- Feature importance ------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Returns feature importances from the trained XGBoost model, sorted descending."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet -- call fit() or load_model() first.")
        model: XGBRegressor = self.pipeline.named_steps["model"]
        importances = model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_columns_, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    from data_processing import generate_full_synthetic_dataset, FeatureEngineer, user_train_val_test_split

    users_df, questions_df, submissions_df = generate_full_synthetic_dataset()
    train_u, val_u, test_u, train_s, val_s, test_s = user_train_val_test_split(users_df, submissions_df)

    fe_train = FeatureEngineer(train_u, questions_df, train_s)
    fe_val = FeatureEngineer(val_u, questions_df, val_s)
    fe_test = FeatureEngineer(test_u, questions_df, test_s)

    train_matrix = fe_train.build_user_feature_matrix()
    val_matrix = fe_val.build_user_feature_matrix()
    test_matrix = fe_test.build_user_feature_matrix()

    predictor = ContestRatingPredictor()
    val_metrics = predictor.fit(train_matrix, val_matrix=val_matrix, use_grid_search=False)
    test_metrics = predictor.evaluate(test_matrix)

    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("\nTop feature importances:")
    print(predictor.feature_importance().head(10).to_string(index=False))

