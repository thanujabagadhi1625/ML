"""
config.py
---------
Single source of truth for constants used across the pipeline.
Centralizing config avoids magic numbers scattered across modules and
makes the system easy to re-tune for a real LeetCode data dump later.
"""

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------------
# Synthetic Data Generation (DATA ENGINE)
# --------------------------------------------------------------------------
NUM_USERS = 500
NUM_QUESTIONS = 300
NUM_SUBMISSIONS = 25_000

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

# --------------------------------------------------------------------------
# Feature Engineering (DATA ENGINE)
# --------------------------------------------------------------------------
RECENCY_HALF_LIFE_DAYS = 21.0   # controls exponential time-decay for momentum
DATASET_WINDOW_DAYS = 365       # submissions span the last N days

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
HYBRID_CF_WEIGHT = 0.7          # blend weight: CF score vs content-similarity score
HYBRID_CONTENT_WEIGHT = 0.3

# --------------------------------------------------------------------------
# Prediction Engine (contest rating regression)
# --------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
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
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [4, 6, 8],
    "model__learning_rate": [0.01, 0.03, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
}
TEST_SIZE = 0.2
CV_FOLDS = 5
