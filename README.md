# LeetCode Mentor: ML-Powered Algorithmic Performance Profiler & Recommendation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4%2B-EB5424.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LeetCode Mentor is an offline-trained, local-first pair-programming mentor that ingests genuine LeetCode submission history to build structured topic weakness profiles, identify semantic failure clusters via sentence transformers, and recommend targeted practice problems through an implicit-feedback matrix factorization (ALS) and content-based hybrid recommender—serving sub-second online inference without retraining.

---

## Overview

Practicing algorithmic coding problems on platforms like LeetCode often suffers from an unstructured, comfort-seeking bias: developers repeatedly solve problems in familiar domains while struggling to detect time-decayed skill degradation, recurring problem statement traps, or conceptual blind spots.

**LeetCode Mentor** transforms raw LeetCode submission event logs into an actionable, privacy-preserving engineering dashboard. Designed around a strict **Offline Training + Online Single-User Inference** architecture, it trains multi-user collaborative filtering and contest rating models offline against a controlled benchmark population, then projects an individual developer's live submission history into the factor space in real time using closed-form linear algebra.

---

## Problem Statement

Standard platform dashboards provide flat aggregate metrics: total solved problems, global acceptance percentages, and calendar heatmaps. These summary statistics fail to guide effective technical interview preparation because:

1. **Lack of Difficulty & Exposure Normalization**: Solving 50 Easy Array problems inflates raw success metrics while masking a 0% success rate on Medium Dynamic Programming problems.
2. **Temporal Skill Blindness**: A topic mastered six months ago can atrophy, but flat lifetime statistics treat past proficiency as identical to current capability.
3. **Coarse-Grained Taxonomic Labels**: Standard tags like "Binary Search" or "Graph" group dozens of distinct problem idioms (e.g., binary search on monotonic answer spaces vs. binary search on sorted matrices).
4. **Uncalibrated Recommendation**: Randomly picking "unsolved medium" problems risks selecting questions misaligned with the user's specific failure points and growth frontiers.

---

## Key Features

- **Local Submission Synchronization**: Ingests genuine LeetCode submission history via a Chrome extension (Manifest V3) or local JSON/CSV/TSV exports directly to a localhost FastAPI server.
- **Strict Offline/Online Decoupling**: Models are trained offline on multi-user populations; live real-user inference runs without retraining, preventing overfitting and latency spikes.
- **Canonical Question Catalogue**: Standardized universe of 453 LeetCode problems spanning 20 canonical algorithmic topics across Easy, Medium, and Hard tiers.
- **Multi-Factor Weakness Profiling**: Computes composite risk scores combining lifetime success rate, time-decayed accuracy, and volume of failed attempts across topics.
- **Exponential Recency Momentum**: Applies time-decay weighting ($0.5^{\Delta t / 21\text{ days}}$) to prioritize recent form over historical performance.
- **Dense NLP Failure Clustering**: Encodes failed problem descriptions into 384-dimensional dense semantic vectors using `sentence-transformers` (`all-MiniLM-L6-v2`), groups them with silhouette-optimized K-Means, and extracts cluster keywords via canonical class-based TF-IDF (c-TF-IDF).
- **Online ALS Closed-Form Fold-In**: Maps a real user into pre-trained implicit ALS latent space in $<1\text{ ms}$ via closed-form ridge regression without modifying catalog item factors.
- **Hybrid Recommendation Engine**: Blends collaborative filtering (50%), tag-based content similarity (30%), and structured weakness boosting (20%) while enforcing catalog diversity.
- **Contest Performance Benchmark Estimate**: Predicts a proxy contest rating on 15 observable behavioral features using an offline-trained XGBoost pipeline.
- **Privacy-First Architecture**: 100% localhost execution. Zero cookies, passwords, or personal submission records are transmitted to third-party endpoints.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Browser Context
        LC[LeetCode Authenticated Session] -->|Fetch Submissions| EXT[Chrome Extension Manifest V3]
    end

    subgraph Local Backend: FastAPI
        EXT -->|POST /api/sync| API[FastAPI Server: localhost:8000]
        UPLOAD[Manual JSON Export] -->|POST /upload| API
        API -->|Persist Locally| STORE[(Local sync_store.json)]
    end

    subgraph Online Single-User Inference
        STORE -->|Load Submissions| NORM[Data Normalization & Cleaning]
        NORM --> FE[Feature Engineering: 15 Observable Features]
        
        FE -->|Vectorized Aggregates| XGB[XGBoost Rating Predictor: Loaded Artifact]
        XGB --> RATING[Contest Benchmark Rating Estimate]
        
        FE -->|Historical Interactions| ALS_FOLD[ALS Closed-Form Fold-In: x_u solve]
        FE -->|Topic Profiles| WEAK[Structured Topic Weakness Scoring]
        FE -->|Tag Vectors| CONTENT[Content-Based Tag Similarity]
        
        ALS_FOLD --> HYBRID[Hybrid Recommendation Engine]
        WEAK --> HYBRID
        CONTENT --> HYBRID
        HYBRID --> RECS[Top-5 Recommended Questions]
        
        NORM -->|Failed Statements| NLP[SentenceTransformer + c-TF-IDF Clustering]
        NLP --> SUBPATTERNS[Semantic Failure Clusters]
    end

    subgraph Local Presentation: Streamlit
        RATING --> DASH[Streamlit Dashboard: localhost:8501]
        WEAK --> DASH
        RECS --> DASH
        SUBPATTERNS --> DASH
    end
```

---

## Offline Training vs. Online Single-User Inference

The core architectural boundary separates population-level offline training from individual online serving:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                   OFFLINE TRAINING                                       │
│                                                                                          │
│   Synthetic Multi-User Population (600 users, 30k events, 9 archetypes)                  │
│                             │                                                            │
│                             ▼                                                            │
│   User-Level Split (70% Train / 15% Val / 15% Test) + Temporal 80/20 Chronological Split │
│                             │                                                            │
│         ┌───────────────────┼────────────────────────┐                                   │
│         ▼                   ▼                        ▼                                   │
│   XGBoost Regressor    Implicit ALS Model     Canonical Catalogue (453 questions)        │
│   (15 observable      (Learns item factors    & Dense Embeddings (all-MiniLM-L6-v2)      │
│    features)           Y and Y^T Y)                                                      │
│         │                   │                        │                                   │
│         ▼                   ▼                        ▼                                   │
│   [rating_model.pkl]   [als_model.npz]        [canonical_questions.json]                 │
│                                               [question_embeddings.npy]                  │
└────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                         │ Artifacts Loaded Read-Only
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              ONLINE SINGLE-USER INFERENCE                                │
│                                                                                          │
│   Real User Submission History (from Chrome Extension / Export)                          │
│                             │                                                            │
│                             ▼                                                            │
│   Feature Extraction (Same pipeline, zero latent variables)                              │
│                             │                                                            │
│         ┌───────────────────┼────────────────────────┐                                   │
│         ▼                   ▼                        ▼                                   │
│   Contest Benchmark    ALS Fold-In             Structured Weakness & NLP Clustering      │
│   Rating Inference     x_u = (Y^T Cu Y+λI)^-1  (c-TF-IDF failure subpatterns)            │
│   (Static Model)        Y^T Cu p_u                   │                                   │
│         │                   │                        │                                   │
│         └───────────────────┼────────────────────────┘                                   │
│                             ▼                                                            │
│               Hybrid Recommendation Ranking & Explanation                                │
│                             │                                                            │
│                             ▼                                                            │
│               Interactive Streamlit Mentor Report                                        │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Why This Separation Is Necessary
1. **Single-User Statistical Limitations**: An individual user solving 50–300 problems does not constitute a multi-user interaction distribution. You cannot train an Alternating Least Squares user-item matrix or a population-level regression model on a sample size of $N=1$.
2. **Overfitting & Latency Prevention**: Retraining complex models on every browser sync would introduce high latency and catastrophic overfitting. Precomputing item factor matrices ($Y \in \mathbb{R}^{453 \times 24}$) and the Gramian ($Y^T Y \in \mathbb{R}^{24 \times 24}$) reduces online serving to solving a single $24 \times 24$ linear system in $<1\text{ ms}$.
3. **Zero Retraining Guarantee**: Model weights remain static and bit-for-bit immutable during dashboard execution.

---

## Data Flow

1. **Ingestion**: The user triggers synchronization via the Chrome extension or uploads an export file. Submissions are sent via `POST /api/sync` to FastAPI and persisted to `files/sync_store.json`.
2. **Normalization**: Submissions are parsed by `_normalize_export_submissions()`, standardizing timestamps to UTC datetime, matching problem titles/slugs against the canonical catalog, and validating status fields.
3. **Feature Engineering**: `FeatureEngineer` executes vectorized Pandas aggregations across difficulties, submission intervals, and time-decayed accuracies to produce a 15-dimensional numeric vector.
4. **Inference Pipelines**:
   - **XGBoost**: Loads `models/rating_model.pkl` and predicts the benchmark rating.
   - **ALS Fold-In**: Forms the user's interaction vector $p_u$ and confidence diagonal $C_u$, solving for the latent user vector $x_u$ against static item factors $Y$.
   - **Topic Weakness Profiling**: Evaluates per-topic failure rates, decayed accuracies, and problem exposure.
   - **NLP Engine**: Gathers failed submissions, computes semantic embeddings, clusters problem representations, and extracts keyword signatures.
5. **Hybrid Blending & Filtering**: Recommender blends normalized scores, strictly filters out already-solved problems, enforces topic diversity (max 2 per tag), and attaches explanatory rationale strings.
6. **Presentation**: Streamlit renders metrics, tables, explanations, and cluster views.

---

## Data Sources

The project maintains a strict boundary between real user data and synthetic benchmark data:

- **Genuine User Submission History**:
  - Ingested locally through the Chrome extension or uploaded export files.
  - Used exclusively at inference time to evaluate performance and generate personal recommendations.
  - Stored locally on `localhost` (`files/sync_store.json`), explicitly ignored by `.gitignore`, and never committed to version control.
- **Synthetic Multi-User Benchmark Population**:
  - Generated via an Item Response Theory (IRT) simulation in `files/data_processing.py`.
  - Used offline to train the population-level collaborative filtering item factors and XGBoost regressor.
  - Incorporates 9 distinct behavioral archetypes with realistic skill progressions, topic preferences, and attempt distributions.

---

## Feature Engineering

The feature pipeline transforms raw submission timestamps and outcomes into 15 observable behavioral features:

| Feature Name | Description | Rationale |
| :--- | :--- | :--- |
| `account_age_days` | Days since user's earliest submission | Practice horizon and overall exposure |
| `total_submissions` | Total cumulative submissions | Overall volume of practice effort |
| `total_accepted` | Total successful submissions | Absolute volume of solved milestones |
| `overall_accuracy` | Ratio of accepted submissions to total | Baseline global accuracy |
| `accuracy_easy` | Success rate on Easy-tier problems | Foundation and syntax fluency |
| `accuracy_medium` | Success rate on Medium-tier problems | Standard interview benchmark capability |
| `accuracy_hard` | Success rate on Hard-tier problems | Advanced algorithmic mastery |
| `share_easy` | Fraction of attempts spent on Easy problems | Practice distribution tier balance |
| `share_medium` | Fraction of attempts spent on Medium problems | Core interview focus allocation |
| `share_hard` | Fraction of attempts spent on Hard problems | Willingness to tackle complex challenges |
| `recency_momentum` | Exponentially time-decayed accuracy | Current form vs. stale historical accuracy |
| `recent_submission_count` | Submission count within last $2\tau$ days | Current activity and practice frequency |
| `unique_problems_attempted`| Count of distinct problem IDs attempted | Breadth of problem catalog coverage |
| `attempts_per_problem` | Total submissions / unique attempted problems | Persistence and debugging repetition |
| `failure_rate` | Ratio of failed submissions to total | Frequency of encountering obstacles |

**Zero Target Leakage Guarantee**: All latent simulation parameters (`latent_skill`, `base_skill`, `hard_resilience`, `learning_slope`, `archetype`) are explicitly excluded from the feature matrix before model training.

---

## Weak Topic Detection

Topic weakness is evaluated using a composite risk formulation that prevents small-sample skew (e.g., failing a single problem 1/1 should not outweigh failing 20 problems at 20% accuracy):

### Mathematical Formulation
For each topic $t$:
$$\text{RiskScore}(t) = 0.50 \cdot (1 - \text{Acc}_{\text{decayed}}) + 0.30 \cdot (1 - \text{Acc}_{\text{lifetime}}) + 0.20 \cdot \min\left(1, \frac{\text{Failures}}{5}\right)$$

where:
- $\text{Acc}_{\text{lifetime}} = \frac{\text{Accepted}_t}{\text{Attempts}_t}$
- $\text{Acc}_{\text{decayed}} = \frac{\sum_{i \in \text{attempts}_t} w_i \cdot \mathbf{1}(\text{status}_i = \text{Accepted})}{\sum_{i \in \text{attempts}_t} w_i}, \quad w_i = 0.5^{\frac{\Delta t_i}{21.0}}$
- $\min\left(1, \frac{\text{Failures}}{5}\right)$ provides an exposure penalty scaling with accumulated failure evidence.

### Weakness Classifications
- **Critical** ($\text{RiskScore} \ge 0.70$): High volume of failures combined with low recent accuracy; immediate remediation required.
- **Weak** ($0.55 \le \text{RiskScore} < 0.70$): Noticeable failure rate or declining recency momentum.
- **Moderate** ($0.40 \le \text{RiskScore} < 0.55$): Balanced performance with occasional struggle on Hard problems.
- **Neutral** ($0.25 \le \text{RiskScore} < 0.40$): Consistent success rate across Medium problems.
- **Strong** ($\text{RiskScore} < 0.25$): High accuracy on Medium and Hard problems with sustained recent momentum.

---

## NLP Failure-Pattern Pipeline

When a user repeatedly fails problems within a broad topic, standard category tags cannot distinguish between failure modes. The secondary NLP pipeline extracts semantic failure patterns from problem statements:

```
Failed Problem Descriptions
            │
            ▼
Sentence-Transformers (all-MiniLM-L6-v2) ──> 384-dimensional dense vectors
            │
            ▼
K-Means Clustering ──> Optimal k ∈ [3, 10] via Silhouette Score sweep
            │
            ▼
Canonical Class-based TF-IDF (c-TF-IDF) ──> Distinctive keyword signatures per cluster
```

### Canonical c-TF-IDF Formulation
Formulated according to Maarten Grootendorst:
$$W_{t, c} = \frac{tf_{t, c}}{w_c} \cdot \ln\left(1 + \frac{A}{tf_t}\right)$$

where:
- $tf_{t, c}$ is the frequency of word $t$ in cluster $c$
- $w_c = \sum_t tf_{t, c}$ is the total word count in cluster $c$
- $A = \frac{1}{K} \sum_c w_c$ is the average word count per cluster across all $K$ clusters
- $tf_t = \sum_c tf_{t, c}$ is the total frequency of word $t$ across all clusters.

### Cold-Start Fallback
If the user has fewer than 5 failed submissions, clustering is skipped and an informative cold-start notice is displayed, avoiding spurious clusters on insufficient evidence.

---

## Hybrid Recommendation System

The recommendation engine combines collaborative filtering, content similarity, and topic weakness boosting:

$$\text{FinalScore}_i = 0.50 \cdot \text{CF}_{\text{norm}}(i) + 0.30 \cdot \text{Content}_{\text{norm}}(i) + 0.20 \cdot \text{WeaknessBoost}(i)$$

### 1. Collaborative Filtering via ALS Fold-In
Trains on implicit feedback interactions ($r_{ui} = 3.0$ for Accepted, $1.0$ for Attempted) with confidence $c_{ui} = 1 + \alpha r_{ui}$ ($\alpha = 15.0$, $\lambda = 20.0$, $d = 24$):
$$x_u = \left(Y^T Y + Y_u^T (C_u - I) Y_u + \lambda I_d\right)^{-1} Y_u^T C_u \mathbf{1}$$
- **Cold-Start Guard**: Requires $\ge 3$ unique attempted problems. If $< 3$, the system smoothly falls back to $0.80 \cdot \text{Content} + 0.20 \cdot \text{Weakness}$.

### 2. Content-Based Tag Similarity
Constructs a user profile vector from solved problems (weight 1.0) and failed problems (weight 0.75) across canonical topic tags, computing cosine similarity against candidate questions:
$$\text{ContentScore}_i = \frac{\mathbf{v}_i \cdot \mathbf{p}_{\text{user}}}{\|\mathbf{v}_i\| \|\mathbf{p}_{\text{user}}\| + \epsilon}$$

### 3. Weakness-Aware Boosting
Candidate questions covering topics flagged as `Critical` or `Weak` receive an additive boost proportional to the topic's risk score.

### Post-Processing & Filtering
- **Solved Problem Deduplication**: Any question previously solved by the user is strictly excluded from recommendations.
- **Topic Diversity**: Enforces a maximum of 2 recommendations per primary topic tag.
- **Explainable Rationale**: Generates transparent reasons explaining why each problem was recommended (e.g., targeted remediation, peer learning path).

---

## Contest Performance Estimation

The contest performance regressor uses an offline-trained **XGBoost Pipeline** (`StandardScaler` + `XGBRegressor`) operating on the 15 observable behavioral features:

- **Hyperparameters**: `n_estimators=400`, `max_depth=5`, `learning_rate=0.03`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_alpha=0.1`, `reg_lambda=1.0`, `objective="reg:squarederror"`.
- **Target Variable**: Continuous synthetic benchmark rating calibrated to the LeetCode contest rating scale $[800, 3000]$.
- **Inference Immutability**: Online predictions execute against static, pre-loaded weights with zero training overhead.

> [!IMPORTANT]
> **Benchmark Proxy Disclaimer**: The predicted contest rating is an offline statistical benchmark proxy trained on a synthetic multi-user population. It is **not** an official LeetCode contest rating.

---

## Synthetic Benchmark

Because LeetCode does not provide an open multi-user dataset of timestamped submission event streams and historical contest ratings, the offline models are trained on a controlled synthetic population:

- **Volume**: 600 synthetic users, 30,000 simulated submission event logs.
- **Question Catalog**: 453 canonical LeetCode problems covering 20 algorithmic taxonomy tags.
- **9 Behavioral Archetypes**:
  1. `strong_overall`: High baseline skill, rapid progression across all problem tiers.
  2. `weak_dp`: Strong general proficiency, but specific deficiency in Dynamic Programming.
  3. `weak_graph`: Proficient in linear structures, struggles with Graph and Tree algorithms.
  4. `strong_easy_med_weak_hard`: Solid Easy/Medium success rate, sharp drop-off on Hard problems.
  5. `improving`: Moderate starting skill with strong positive learning slope over time.
  6. `declining`: High initial skill followed by inactivity and decay.
  7. `stable`: Consistent middle-tier performance over long practice windows.
  8. `high_attempt_low_accuracy`: High attempt volume with low initial success rate; brute-force practice behavior.
  9. `specialized`: High proficiency in Arrays/Strings/Math, low exposure to advanced structures.
- **User-Level Partitioning**: 70% Train (420 users) | 15% Validation (90 users) | 15% Test (90 users). No user appears in multiple splits.
- **Temporal Split**: Submissions per test user are chronologically split (80% observation history, 20% held-out future).

---

## Offline Evaluation Metrics

All models were evaluated on strictly held-out test users (15% split) and future temporal interactions (20% split). The authoritative evaluation metrics serialized in `models/model_metadata.json` are:

### 1. Contest Rating Regressor (XGBoost)
| Metric | Test Score | Description |
| :--- | :--- | :--- |
| **RMSE** | `177.29` | Root Mean Squared Error on held-out test users |
| **MAE** | `136.23` | Mean Absolute Error in rating points |
| **$R^2$** | `0.430` | Variance explained over test user distribution |

*Note on $R^2 \approx 0.430$*: Human contest performance exhibits high stochastic variance (unfamiliar problem idioms, implementation bugs, time pressure). An $R^2 = 0.430$ demonstrates that the model captures strong structural signal from practice history without overfitting to synthetic noise.

### 2. Hybrid Recommendation Engine
| Metric | Test Score | Random Baseline | Improvement |
| :--- | :--- | :--- | :--- |
| **Hit Rate@5** | `8.99%` | `4.65%` | **$1.93\times$** over random |
| **Precision@5**| `1.80%` | `0.95%` | **$1.90\times$** over random |
| **Recall@5**   | `2.17%` | — | Captured fraction of future solved problems |
| **NDCG@5**     | `0.0219`| — | Normalized DCG bounded by $\min(K, \|\mathcal{G}_u\|)$ |

### 3. Weak-Topic Detection
| Metric | Test Score | Description |
| :--- | :--- | :--- |
| **Precision** | `62.40%` | Precision in predicting future topic failure points |
| **Recall** | `49.62%` | Recall in capturing future topic failure points |
| **F1 Score** | `0.5056` | Harmonic mean of weak-topic precision & recall |

---

## FastAPI Backend Server

The local backend in `files/api_server.py` exposes the following endpoints on `http://127.0.0.1:8000`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Returns local server status, count of synced accounts, and last sync timestamp. |
| `POST` | `/api/sync` | Receives submission batches from the Chrome extension, deduplicates by `(problem_slug, timestamp)`, and updates `sync_store.json`. |
| `POST` | `/upload` | Accepts manual JSON export file uploads from the browser. |
| `POST` | `/fetch` | Direct session-cookie fetch helper for localhost development scripting. |

---

## Chrome Extension

The extension in `extension/` provides a zero-setup synchronization interface:
- **Manifest V3**: Modern background service worker and content script configuration.
- **Authenticated Local Fetch**: Uses the user's active `leetcode.com` session to query submission history via GraphQL/REST.
- **Sync Modes**:
  - *Sync New*: Incrementally fetches recent submissions without overwriting existing records.
  - *Rebuild History*: Fetches full submission history up to the selected limit (100, 500, 1000).
- **Direct POST**: Sends normalized submission payloads directly to `http://127.0.0.1:8000/api/sync`.

---

## Streamlit Dashboard

The interactive user interface in `files/streamlit_app.py` provides:
1. **Data Source Selector**: Choose between Chrome Extension Sync, Uploaded Export (JSON/CSV/TSV), or Demo Dataset (`large_user.json`).
2. **Data Provenance**: Reports active username, submission count, data source, and synchronization timestamp.
3. **Data Quality Diagnostics**: Integrity check tracking duplicate records, missing fields, and date ranges.
4. **Topic Weakness Matrix**: Interactive table showing topic attempts, success rate %, exposure tier, dominant difficulty, and weakness classification.
5. **Actionable Explanations**: Diagnostic cards explaining root causes for flagged topics (e.g., declining recency momentum, low accuracy on Medium problems).
6. **Top-5 Recommendations**: Ranked practice problems with difficulty, topic tags, recommendation score, and explicit rationale.
7. **Secondary NLP Subpatterns**: Expandable view showing semantically clustered problem failure patterns with extracted c-TF-IDF keyword signatures.
8. **Contest Rating Model Status**: Reports offline test metrics (RMSE, MAE, $R^2$) and benchmark disclaimer.
9. **Raw JSON Inspector**: Full inspectable report payload for debugging and verification.

---

## Tech Stack

- **Machine Learning & Core Analytics**: Python 3.10+, NumPy, Pandas, Scipy, Scikit-Learn, PyTorch, Sentence-Transformers (`all-MiniLM-L6-v2`), XGBoost, Joblib.
- **Backend API**: FastAPI, Uvicorn, Pydantic v2, Requests.
- **Frontend Dashboard**: Streamlit.
- **Browser Extension**: JavaScript (ES6+), Chrome Manifest V3, HTML5, CSS3.
- **Testing & Verification**: Unittest (31 tests).

---

## Project Structure

```
.
├── .gitignore                      # Comprehensive privacy & runtime ignore rules
├── .streamlit/
│   └── config.toml                 # Streamlit UI runtime configuration
├── LICENSE                         # MIT Open Source License
├── README.md                       # Complete technical documentation
├── requirements.txt                # Pinned dependencies
├── extension/                      # Chrome Extension (Manifest V3)
│   ├── content.js                  # Authenticated LeetCode session scraper
│   ├── manifest.json               # Extension manifest & permissions
│   ├── popup.html                  # Extension popup UI
│   ├── popup.js                    # Extension controller & API dispatcher
│   └── styles.css                  # Dark-theme popup styling
├── models/                         # Offline-Trained Static Model Artifacts
│   ├── als_model.npz               # Trained ALS item factors Y and Gramian Y^T Y
│   ├── canonical_questions.json    # 453 canonical questions across 20 topics
│   ├── model_metadata.json         # Authoritative hyperparameters & test metrics
│   ├── question_embeddings.npy     # Precomputed sentence embeddings (453, 384)
│   └── rating_model.pkl            # Pre-trained XGBoost contest rating pipeline
├── training/                       # Offline Training & Evaluation Scripts
│   ├── __init__.py
│   ├── build_canonical_catalogue.py # Generates canonical 453-question catalog
│   ├── evaluate_models.py          # Standalone benchmark evaluation runner
│   └── train_models.py             # Offline training orchestrator
├── files/                          # Online Inference & Application Source
│   ├── api_server.py               # Local FastAPI synchronization server
│   ├── config.py                   # Central constants & path configuration
│   ├── data_processing.py          # Vectorized feature engineering & simulation
│   ├── export_helper.py            # CLI utilities for export file parsing
│   ├── leetcode_fetcher.py         # Session-based submission fetcher
│   ├── main.py                     # LeetCodeMentor inference orchestrator
│   ├── nlp_cluster.py              # SentenceTransformers + c-TF-IDF clustering
│   ├── predictor.py                # XGBoost ContestRatingPredictor
│   ├── recommender.py              # ALSMatrixFactorization & HybridRecommender
│   ├── schemas.py                  # Pydantic schema definitions
│   ├── streamlit_app.py            # Interactive Streamlit dashboard UI
│   └── demo/
│       └── large_user.json         # Synthetic demo dataset for local testing
└── tests/                          # Unit & Integration Test Suite (31 tests)
    ├── __init__.py
    ├── test_api_server.py          # FastAPI sync & status endpoint tests
    ├── test_data_processing.py     # Archetypes, user splits, temporal splits & features
    ├── test_end_to_end.py          # End-to-end inference & zero-retraining verification
    ├── test_nlp_cluster.py         # Embedding, clustering, c-TF-IDF & cold-start guards
    ├── test_predictor.py           # Offline XGBoost, serialization & feature importance
    └── test_recommender.py         # ALS fold-in, cold-start fallback & diversity rules
```

---

## Local Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Google Chrome or Chromium-based browser
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/thanujabagadhi1625/ML.git
cd ML

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (cmd.exe):
.\.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

### 1. Start the Local FastAPI Backend
In your first terminal window:
```bash
cd files
uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```
*Or alternatively:*
```bash
python files/api_server.py
```
The server will start at `http://127.0.0.1:8000`. Verify by checking `http://127.0.0.1:8000/api/status`.

### 2. Load the Chrome Extension
1. Open Google Chrome and navigate to `chrome://extensions`.
2. Toggle **Developer mode** in the top-right corner.
3. Click **Load unpacked**.
4. Select the `extension/` directory from the root of this repository.
5. The **LeetCode Mentor Sync** icon will appear in your browser toolbar.

### 3. Synchronize Submissions
1. Open [https://leetcode.com](https://leetcode.com) and ensure you are logged in.
2. Click the extension icon in your browser toolbar.
3. Select a fetch limit and click **Sync New** or **Rebuild History**.
4. The extension transfers your submissions directly to your local FastAPI server.

### 4. Start the Streamlit Dashboard
In a second terminal window (with virtual environment activated):
```bash
streamlit run files/streamlit_app.py
```
The dashboard opens automatically in your browser at `http://localhost:8501`. Select **My LeetCode Data (Extension Sync)** and click **Generate Report**.

*(Optional: If you do not have a live LeetCode account, select **Demo Dataset** to explore all features using `files/demo/large_user.json`.)*

---

## Testing

Execute the complete unit and integration test suite:
```bash
python -m unittest discover tests
```
*Expected result:*
```text
Ran 31 tests in ~40s
OK
```

---

## (Optional) Retraining Offline Models

The repository comes with pre-trained artifacts in `models/`. To regenerate the canonical catalog or retrain the benchmark models from scratch:
```bash
# 1. Regenerate canonical 453-question universe
python training/build_canonical_catalogue.py

# 2. Train XGBoost, ALS, and precompute dense embeddings
python training/train_models.py

# 3. Run standalone benchmark evaluation
python training/evaluate_models.py
```

---

## Privacy & Security

This project implements a strict local-first privacy model:
- **No Remote Credential Transmission**: The Chrome extension communicates **exclusively** with `http://127.0.0.1:8000`. No session tokens, cookies, or submission records are sent to cloud servers.
- **Local File Persistence**: Synced data is saved locally to `files/sync_store.json`. This file is explicitly listed in `.gitignore` and is never committed to Git.
- **No Hardcoded Secrets**: Zero API keys, passwords, or personal credentials exist in the codebase.
- **Sanitized Demo Data**: The repository includes only algorithmic synthetic demo fixtures (`files/demo/large_user.json`).

---

## Limitations

1. **Synthetic Population Domain Gap**: Offline models are trained on simulated Item Response Theory distributions. While behavioral archetypes mirror human practice patterns, synthetic data cannot replicate all human nuances (e.g., contest server outages, copying external solutions).
2. **Benchmark Proxy Rating**: The contest rating regressor estimates expected performance on a benchmark scale and should not be confused with official LeetCode contest ratings.
3. **Cold-Start Boundary for ALS**: The closed-form fold-in requires $\ge 3$ unique attempted questions to construct a stable collaborative vector. Users with fewer interactions rely on content-based similarity and weakness boosting.
4. **Catalog Scope**: Problem recommendations and semantic search operate within the 453-question canonical catalog. Questions outside this set are aligned via slug matching or topic tag projection.

---

## Future Improvements

- [ ] AST parsing of submitted Python/C++ solutions to detect algorithmic antipatterns and asymptotic time complexity bugs.
- [ ] Spaced-repetition scheduling (e.g., SuperMemo SM-2) for review intervals on historically failed problems.
- [ ] Support for Codeforces and HackerRank submission schema ingestion.
- [ ] Local quantized LLM integration (via Ollama/llama.cpp) for generating contextual hints without revealing full solutions.

---

## Author

**Thanuja Bagadhi**  
Department of Computer Science & Engineering  
National Institute of Technology Rourkela (NIT Rourkela)  
GitHub: [https://github.com/thanujabagadhi1625](https://github.com/thanujabagadhi1625)  

---

## License

This project is licensed under the [MIT License](LICENSE).
