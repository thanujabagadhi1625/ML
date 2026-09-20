"""
config.py
---------
Single source of truth for constants used across the pipeline.
Centralizing config avoids magic numbers scattered across modules and
makes the system easy to re-tune for a real LeetCode data dump later.
"""

from pathlib import Path
import numpy as np

# --------------------------------------------------------------------------
# Paths & Artifact Locations
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
CANONICAL_QUESTIONS_PATH = MODELS_DIR / "canonical_questions.json"
RATING_MODEL_PATH = MODELS_DIR / "rating_model.pkl"
ALS_MODEL_PATH = MODELS_DIR / "als_model.npz"
QUESTION_EMBEDDINGS_PATH = MODELS_DIR / "question_embeddings.npy"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

import pandas as pd

# --------------------------------------------------------------------------
# Reproducibility & Profiles
# --------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
REFERENCE_DATE = pd.Timestamp("2026-01-01").normalize()

CONFIG_PROFILES = {
    "v1": {
        "NUM_USERS": 600,
        "NUM_QUESTIONS": 453,
        "NUM_SUBMISSIONS": 30_000,
        "GENERATOR_VERSION": "v1",
    },
    "v3": {
        "NUM_USERS": 2000,
        "NUM_QUESTIONS": None,
        "NUM_SUBMISSIONS": None,
        "GENERATOR_VERSION": "v3",
        "BETA": 1.5,
        "P_RETRY": 0.35,
    },
}

CURRENT_PROFILE = "v1"

def set_profile(name: str):
    global CURRENT_PROFILE, NUM_USERS, NUM_QUESTIONS, NUM_SUBMISSIONS
    if name not in CONFIG_PROFILES:
        raise ValueError(f"Unknown config profile: {name}")
    CURRENT_PROFILE = name
    p = CONFIG_PROFILES[name]
    if p.get("NUM_USERS") is not None:
        NUM_USERS = p["NUM_USERS"]
    if p.get("NUM_QUESTIONS") is not None:
        NUM_QUESTIONS = p["NUM_QUESTIONS"]
    if p.get("NUM_SUBMISSIONS") is not None:
        NUM_SUBMISSIONS = p["NUM_SUBMISSIONS"]

# --------------------------------------------------------------------------
# Synthetic Data Generation & Archetypes (DATA ENGINE)
# --------------------------------------------------------------------------
NUM_USERS = 600
NUM_QUESTIONS = 453
NUM_SUBMISSIONS = 30_000

USER_ARCHETYPES = [
    "strong_overall",
    "weak_dp",
    "weak_graph",
    "strong_easy_med_weak_hard",
    "improving",
    "declining",
    "stable",
    "high_attempt_low_accuracy",
    "specialized",
]

DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFFICULTY_WEIGHTS = [0.35, 0.45, 0.20]
DIFFICULTY_SCORE = {"Easy": 1.0, "Medium": 2.2, "Hard": 3.6}  # used to drive synthetic solve-probability

TOPIC_TAGS = [
    "Array", "String", "Hash Table", "Dynamic Programming", "Math",
    "Sorting", "Greedy", "Depth-First Search", "Breadth-First Search",
    "Binary Search", "Tree", "Graph", "Two Pointers", "Sliding Window",
    "Backtracking", "Bit Manipulation", "Heap", "Stack", "Trie", "Union Find",
]

SUBMISSION_STATUSES = ["Accepted", "Wrong Answer", "Time Limit Exceeded", "Runtime Error", "Compile Error"]
STATUS_BASE_WEIGHTS = np.array([0.45, 0.30, 0.12, 0.09, 0.04])

LANGUAGES = ["Python3", "C++", "Java", "JavaScript", "Go"]

QUESTION_COLUMNS = ["question_id", "title", "description", "difficulty", "topic_tags", "acceptance_rate"]
USER_COLUMNS = ["user_id", "account_age_days", "latent_skill", "contest_rating"]
SUBMISSION_COLUMNS = ["user_id", "question_id", "timestamp", "status", "runtime_ms", "language"]

# --------------------------------------------------------------------------
# Feature Engineering (DATA ENGINE)
# --------------------------------------------------------------------------
RECENCY_HALF_LIFE_DAYS = 21.0   # controls exponential time-decay for momentum
DATASET_WINDOW_DAYS = 365       # submissions span the last N days

# --------------------------------------------------------------------------
# Data Splitting (User-Level & Temporal)
# --------------------------------------------------------------------------
TRAIN_USER_RATIO = 0.70
VAL_USER_RATIO = 0.15
TEST_USER_RATIO = 0.15
TEMPORAL_HISTORY_RATIO = 0.80   # first 80% interactions as input, last 20% held-out ground truth

# --------------------------------------------------------------------------
# NLP / Clustering Engine (WEAK TOPICS ENGINE)
# --------------------------------------------------------------------------
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384              # native output dim of all-MiniLM-L6-v2
MIN_CLUSTERS = 3
MAX_CLUSTERS = 10
MIN_FAILED_SUBMISSIONS_FOR_CLUSTERING = 5  # cold-start guard
TOP_KEYWORDS_PER_CLUSTER = 6

# --------------------------------------------------------------------------
# Recommendation Engine
# --------------------------------------------------------------------------
MF_LATENT_DIM = 24
MF_REG_LAMBDA = 20.0            # ALS regularization (lambda * I)
MF_CONFIDENCE_ALPHA = 15.0      # implicit-feedback confidence scaling
MF_EPOCHS = 15
TOP_N_RECOMMENDATIONS = 5

# Recommendation score blending weights (tuned on validation users; see results/tuning.json and results/chosen_config.json)
HYBRID_CF_WEIGHT = 0.50          # collaborative filtering weight
HYBRID_CONTENT_WEIGHT = 0.30     # content similarity weight
HYBRID_WEAKNESS_WEIGHT = 0.20    # topic weakness boost weight

# --------------------------------------------------------------------------
# Prediction Engine (contest rating regression)
# --------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
XGB_PARAM_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.03, 0.08],
}
TEST_SIZE = 0.15
CV_FOLDS = 5
