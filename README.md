# LeetCode Mentor: ML-Powered Algorithmic Performance Profiler & Recommendation System

> A technically defensible, privacy-first pair-programming mentor that ingests real LeetCode submission history to build structured topic weakness profiles, extract semantic failure patterns with sentence transformers, and recommend targeted practice problems via offline-trained collaborative and content-based filtering.

---

## Overview

Practicing algorithmic coding problems on platforms like LeetCode often suffers from an unstructured approach: developers repeatedly solve comfortable problems while struggling to identify specific algorithmic weaknesses, time-decayed skill degradation, or recurring problem failure modes.

**LeetCode Mentor** provides an automated, local-first profiling and recommendation system built on an **offline-trained + online single-user inference** architecture:
- **Offline Training**: Multi-user synthetic populations (600 users, 30,000 submission logs across 9 realistic user archetypes) train an XGBoost contest rating regressor, an implicit-feedback Alternating Least Squares (ALS) recommender, and precompute dense problem embeddings using `sentence-transformers` over a canonical question universe (453 problems).
- **Online Single-User Serving**: The local user syncs their submission history via a Chrome extension (Manifest V3) and FastAPI backend. The system serves recommendations and skill evaluations **without retraining**: real user interactions are folded into the fixed ALS item factor space via closed-form ridge regression ($x_u = (Y^T C_u Y + \lambda I)^{-1} Y^T C_u p_u$), while the pre-trained XGBoost pipeline predicts skill rating on 15 observable behavioral features in sub-millisecond time.

---

## Key Features

- **Automated Local Sync**: Ingests LeetCode submission history incrementally or via full history rebuild using a local-only Chrome extension (Manifest V3) and FastAPI sync server.
- **Canonical Question Catalogue**: 453 canonical LeetCode problems spanning 20 algorithmic taxonomy tags across Easy, Medium, and Hard difficulties, ensuring seamless alignment between offline training and real-user submissions.
- **Structured Topic Weakness Profiling**: Evaluates user proficiency across canonical algorithmic topics (e.g., Dynamic Programming, Graph Traversal, Binary Search, Trees) using empirical success rates, problem exposure levels, and difficulty weighting.
- **Time-Decayed Recency Momentum**: Computes exponential time-decayed accuracy ($w_i = 0.5^{\Delta t / \tau}$, $\tau = 21\text{ days}$) to reward current problem-solving capability over historical performance.
- **Secondary NLP Subpattern Discovery**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate dense semantic embeddings of failed problem statements, clusters them with K-Means (optimal $k$ selected via silhouette score), and extracts descriptive keywords using **canonical class-based TF-IDF (c-TF-IDF)**.
- **Zero-Retraining Online Recommendation**: Real-user interactions index into pre-trained ALS item factors via closed-form fold-in, blended with tag-based content similarity and structured topic weakness boosting.
- **Offline-Trained Contest Rating Regression**: Pre-trained XGBoost pipeline estimates skill rating from observable user features without retraining or heuristics masquerading as ML.
- **Privacy-First Architecture**: Zero external telemetry, credentials, or submission logs are sent to cloud servers; all data processing and storage remain on `localhost`.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   LeetCode (Browser Context)                     │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Direct Authenticated Fetch
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             Chrome Extension (Manifest V3 / Popup)               │
│   - Extracts submissions incrementally / rebuild mode            │
│   - Normalizes payload & POSTs to local FastAPI backend          │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ HTTP POST (http://127.0.0.1:8000/api/sync)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             FastAPI Backend Server (files/api_server.py)         │
│   - Validates schema with Pydantic                               │
│   - Persists submission history locally (sync_store.json)        │
│   - Aligns slugs against canonical question catalogue            │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             Offline Training vs. Online Single-User Serving      │
│                                                                  │
│  [OFFLINE TRAINING] (training/train_models.py)                   │
│  - Multi-user synthetic population (600 users, 9 archetypes)     │
│  - User-level split (70/15/15) + temporal history split (80/20)  │
│  - Precompute question embeddings (models/question_embeddings)   │
│  - Train XGBoost Rating Regressor (models/rating_model.pkl)      │
│  - Train ALS Item Factors Y & Y^T Y (models/als_model.npz)       │
│                                                                  │
│  [ONLINE SINGLE-USER SERVING] (files/main.py, streamlit_app.py)  │
│  - Ingests real user submissions & computes 15 observable feats  │
│  - Predicts rating via loaded XGBoost (NO retraining)            │
│  - Closed-form ALS fold-in: x_u = (Y^T Cu Y + λI)^-1 Y^T Cu p_u  │
│  - Hybrid blend: CF (50%) + Content (30%) + Weakness Boost (20%) │
│  - SentenceTransformer + c-TF-IDF failure cluster analysis       │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             Streamlit Dashboard (files/streamlit_app.py)         │
│   - Visual diagnostic summary & data provenance                  │
│   - Tabular topic weakness matrix with actionable explanations   │
│   - Top-5 profile-guided recommendations with transparent reason │
│   - Benchmark rating with model test metrics and honest caveats  │
│   - Expandable secondary NLP subpattern clusters                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Machine Learning & Algorithmic Methodology

### 1. Offline Multi-User Population & Non-Leaking Splits
The training dataset is generated via a simulation engine (`files/data_processing.py`) incorporating 9 realistic programmer archetypes (`strong_overall`, `weak_dp`, `weak_graph`, `strong_easy_med_weak_hard`, `improving`, `declining`, `stable`, `high_attempt_low_accuracy`, `specialized`):
- **User-Level Split**: 600 users partitioned into Train (70%, 420 users), Validation (15%, 90 users), and Test (15%, 90 users). No user appears in multiple splits.
- **Temporal History Split**: For recommender and weak-topic evaluation, user submission histories are chronologically partitioned into an 80% observation window (input) and a 20% future held-out window (ground truth).
- **Observable Feature Boundary**: Feature engineering strictly excludes simulation variables (`latent_skill`, `archetype`, target `contest_rating`).

### 2. Feature Engineering & Recency Momentum
The feature pipeline computes 15 observable aggregates without per-row loops:
- **Difficulty-Specific Accuracy**: Accepted ratios for `Easy`, `Medium`, and `Hard` problems.
- **Difficulty Distribution**: Proportion of total practice effort spent per difficulty tier.
- **Exponential Recency Momentum**:
  $$\text{momentum} = \frac{\sum_i w_i \cdot \mathbb{I}(\text{status}_i = \text{Accepted})}{\sum_i w_i}, \quad w_i = 0.5^{\frac{\text{days\_ago}_i}{\tau}}$$
  where $\tau = 21.0\text{ days}$.

### 3. Weak Topics NLP Clustering & Canonical c-TF-IDF
For failed problem statements:
1. **Semantic Embeddings**: Encodes problem descriptions into 384-dimensional dense vectors using `sentence-transformers` (`all-MiniLM-L6-v2`) with unit-norm normalization.
2. **Optimal Cluster Selection**: Sweeps cluster count $k \in [3, 10]$ and selects the optimal $k$ maximizing the **Silhouette Score**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   with robust edge-case degradation (triggers cold-start note if $<5$ failed submissions).
3. **Canonical Class-based TF-IDF (c-TF-IDF)**: Formulated according to Maarten Grootendorst:
   $$W_{t, c} = \frac{tf_{t, c}}{w_c} \cdot \ln\left(1 + \frac{A}{tf_t}\right)$$
   where:
   - $tf_{t, c}$ is the frequency of word $t$ in cluster $c$
   - $w_c = \sum_t tf_{t, c}$ is the total word count in cluster $c$
   - $A = \frac{1}{K} \sum_c w_c$ is the average word count per cluster across all $K$ clusters
   - $tf_t = \sum_c tf_{t, c}$ is the total frequency of word $t$ across all clusters.

### 4. Hybrid Recommendation Engine & ALS Fold-In
- **Implicit ALS Matrix Factorization**: Trains implicit-feedback ALS factorizing interaction matrix $R \approx X Y^T$ ($d=24$, $\lambda=20.0$, $\alpha=15.0$). Item factors $Y \in \mathbb{R}^{M \times d}$ and precomputed $Y^T Y$ are serialized to `models/als_model.npz`.
- **Online Fold-In for New Users**:
  When a single real user arrives with interaction vector $p_u$ and confidence diagonal $C_u$, their latent vector $x_u$ is solved in closed-form without modifying item factors or retraining:
  $$x_u = \left(Y^T Y + Y_u^T (C_u - I) Y_u + \lambda I\right)^{-1} Y_u^T C_u p_u$$
  where $Y_u$ contains only the item rows corresponding to problems attempted by user $u$.
- **Cold-Start Guard**: Users with $<3$ interactions gracefully fall back to content-based tag similarity and weakness profiling.
- **Normalized Hybrid Blending**:
  $$\text{Score} = 0.50 \cdot \text{ALS}_{\text{norm}} + 0.30 \cdot \text{Content}_{\text{norm}} + 0.20 \cdot \text{WeaknessBoost}$$
- **Post-Processing Rules**:
  - Excludes already solved problems.
  - Enforces topic diversity (maximum 2 recommendations per topic).
  - Generates transparent, human-readable explanations grounded in the user's weakness profile.

### 5. Contest Rating Regression Pipeline
- Pre-trained XGBoost regressor pipeline with `StandardScaler` trained on 15 observable features.
- Serves single-user online inference without modifying pipeline parameters or performing on-the-fly retraining.

---

## Offline Evaluation Metrics (Test Users)

All models are evaluated offline on strictly held-out, untouched test users and serialized to `models/model_metadata.json`:

| Model / Subsystem | Metric | Test Score | Description |
|---|---|---|---|
| **Contest Rating Regressor** | **RMSE** | `177.29` | Root Mean Squared Error on held-out test users |
| | **MAE** | `136.23` | Mean Absolute Error in rating points |
| | **$R^2$** | `0.430` | Variance explained over test user distribution |
| **Recommendation Engine** | **Hit Rate@5** | `8.99%` | Held-out future problems captured in Top-5 recs |
| | **NDCG@5** | `0.0219` | Normalized Discounted Cumulative Gain at Top-5 (strict IDCG@K) |
| | **Precision@5** | `1.80%` | Fraction of recommended items solved in future |
| | **Recall@5** | `2.17%` | Fraction of future solved items captured in recs |
| **Weak-Topic Detection** | **Precision** | `62.40%` | Precision in predicting future topic failure points |
| | **Recall** | `49.62%` | Recall in capturing future topic failure points |
| | **F1 Score** | `0.5056` | Harmonic mean of weak-topic precision & recall |

---

## Tech Stack

| Domain | Technologies & Libraries |
|---|---|
| **Core ML & NLP** | Python 3.10+, NumPy, Pandas, Scipy, Scikit-Learn, PyTorch, Sentence-Transformers, XGBoost, Joblib |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2, Python-Multipart, Requests |
| **Frontend UI** | Streamlit |
| **Browser Extension** | JavaScript (ES6+), Chrome Manifest V3, HTML5, CSS3 |
| **Testing & Tooling** | Unittest, Git |

---

## Project Structure

```
.
├── .gitignore               # Comprehensive ignore rules (secrets, venv, local data)
├── .streamlit/
│   └── config.toml          # Streamlit runtime configuration
├── LICENSE                  # MIT Open Source License
├── README.md                # Technical documentation
├── requirements.txt         # Pinned runtime dependencies
├── extension/               # Chrome Extension (Manifest V3)
│   ├── manifest.json        # Extension configuration & host permissions
│   ├── content.js           # Content script for LeetCode DOM/REST/GraphQL sync
│   ├── popup.html           # Extension popup interface
│   ├── popup.js             # Extension controller & status polling
│   └── styles.css           # Modern dark-theme popup styling
├── models/                  # Offline-Trained Model Artifacts & Catalogue
│   ├── canonical_questions.json    # 453 canonical questions across 20 topics
│   ├── question_embeddings.npy     # Precomputed sentence-transformer embeddings (453, 384)
│   ├── rating_model.pkl            # Pre-trained XGBoost rating pipeline
│   ├── als_model.npz               # Trained ALS item factors Y & precomputed Y^T Y
│   └── model_metadata.json         # Training provenance, hyperparameters & evaluation metrics
├── training/                # Offline Training & Evaluation Scripts
│   ├── build_canonical_catalogue.py # Builds canonical 453-question JSON universe
│   ├── train_models.py             # Offline training orchestrator (runs splits, XGBoost, ALS)
│   └── evaluate_models.py          # Benchmark evaluation suite across test users
├── files/                   # Online Inference & Backend Source Code
│   ├── api_server.py        # Local FastAPI sync server
│   ├── config.py            # Central pipeline constants & artifact paths
│   ├── data_processing.py   # Vectorized feature engineering & realistic archetypes
│   ├── export_helper.py     # CLI utilities for LeetCode export parsing
│   ├── leetcode_fetcher.py  # Session-based fetch helpers
│   ├── main.py              # LeetCodeMentor facade & online inference orchestrator
│   ├── nlp_cluster.py       # SentenceTransformers + K-Means + canonical c-TF-IDF
│   ├── predictor.py         # ContestRatingPredictor (save, load, predict)
│   ├── recommender.py       # ALSMatrixFactorization (fit, save, load, fold-in) + Hybrid
│   ├── schemas.py           # Typed schema contracts for Question/User/Submission
│   ├── streamlit_app.py     # Interactive mentor dashboard UI
│   └── demo/
│       └── large_user.json  # Synthetic demo dataset for testing/evaluation
└── tests/                   # Unit & Integration Test Suite (31 tests)
    ├── __init__.py
    ├── test_api_server.py       # FastAPI sync & status endpoint tests
    ├── test_data_processing.py  # Archetypes, user splits, temporal splits & feature tests
    ├── test_end_to_end.py       # Integration tests & zero-retraining inference guarantee
    ├── test_nlp_cluster.py      # Embedding, clustering, c-TF-IDF & cold-start guards
    ├── test_predictor.py        # Offline XGBoost, serialization & inference without retraining
    └── test_recommender.py      # ALS fold-in for new user, cold-start fallback & diversity
```


---

## Local Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Google Chrome or Chromium-based browser

### 2. Clone Repository & Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/leetcode-mentor.git
cd leetcode-mentor

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (cmd.exe):
.\.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Unit & Integration Test Suite
```bash
python -m unittest discover tests
```
*All 31 unit and integration tests should pass.*

### 4. (Optional) Re-Train Offline Models
The repository includes pre-trained model artifacts in `models/`. If you wish to retrain from scratch or regenerate the synthetic benchmark population:
```bash
python training/train_models.py
```
This script runs the full pipeline:
- Generates 600 synthetic users across 9 archetypes
- Executes non-leaking user-level splits (70/15/15) and temporal history splits (80/20)
- Precomputes question embeddings (`models/question_embeddings.npy`)
- Trains and evaluates XGBoost (`models/rating_model.pkl`)
- Trains ALS item factors ($Y$, $Y^T Y$) (`models/als_model.npz`)
- Serializes evaluation metrics and training metadata (`models/model_metadata.json`)

---

## Running the Application

### Step 1: Start the Local FastAPI Backend Server
In your activated terminal:
```bash
python files/api_server.py
```
The API server starts at `http://127.0.0.1:8000`. It provides:
- `GET /api/status`: Check sync server status and stored submission count.
- `POST /api/sync`: Receive submissions from the Chrome extension.
- `POST /upload`: Upload manual export files.

### Step 2: Load the Chrome Extension
1. Open Google Chrome and navigate to `chrome://extensions`.
2. Toggle **Developer mode** in the top-right corner.
3. Click **Load unpacked**.
4. Select the `extension/` directory from this project root.
5. The **LeetCode Mentor Sync** icon will appear in your browser toolbar.

### Step 3: Synchronize Your Submissions
1. Open [https://leetcode.com](https://leetcode.com) in Chrome and ensure you are logged into your account.
2. Click the **LeetCode Mentor Sync** extension icon in your browser toolbar.
3. Choose a fetch limit (e.g., 100, 500, 1000) and click:
   - **Sync New**: Incrementally fetches new submissions without overwriting existing history.
   - **Rebuild History**: Rebuilds your local stored dataset with the latest submissions up to the selected limit.
4. The extension fetches your submission list via authenticated REST/GraphQL calls and transfers them directly to your local FastAPI server.

### Step 4: Launch the Streamlit Dashboard
In a separate terminal window (with virtual environment activated):
```bash
streamlit run files/streamlit_app.py
```
The dashboard opens in your browser at `http://localhost:8501`.

---

## How to Use the Streamlit Dashboard

1. **Select Data Source**:
   - **My LeetCode Data (Extension Sync)**: Automatically loads the submission records synced via the Chrome extension.
   - **Upload LeetCode Export**: Allows uploading a manual JSON/CSV/TSV submission file.
   - **Demo Dataset**: Loads the built-in synthetic benchmark dataset (`files/demo/large_user.json`) to explore all features without needing a live LeetCode account.
2. **Click "Generate Report"**: The application executes feature extraction, topic weakness analysis, NLP failure clustering, and hybrid recommendations.
3. **Inspect the Results**:
   - **Diagnostic Metrics**: Solved vs. failed submission counts, submission frequency, and benchmark-estimated rating.
   - **Topic Weakness Matrix**: Interactive table showing topic attempts, success rate %, exposure level, and flagged weakness levels (`Critical`, `Weak`, `Moderate`, `Strong`).
   - **Actionable Explanations**: "Why Weakest Topics are Flagged" explains root causes (e.g., declining momentum, high failure rate on Medium/Hard problems).
   - **Top-5 Recommendations**: Curated practice problems with score breakdown, tags, difficulty, and an explicit reason grounded in your weakness profile.
   - **Secondary NLP Subpatterns**: Expandable view showing semantically clustered problem failure patterns with extracted c-TF-IDF keyword signatures.
   - **Contest Rating Model Status**: Reports offline test metrics (RMSE, MAE, $R^2$) and benchmark disclaimer.

---

## Privacy & Security Architecture

This repository is designed with a strict **local-first privacy model**:

- **No Remote Credential Transmission**: The Chrome extension operates inside the user's active browser session on `leetcode.com` and communicates **exclusively** with `http://127.0.0.1:8000`. No cookies, tokens, or submission data are ever transmitted to third-party endpoints.
- **Local Persistence Only**: Synced submission records are written to a local file (`files/sync_store.json`), which is explicitly ignored by `.gitignore` and never committed to version control.
- **Sanitized Demo Data**: The repository includes only synthetic fixtures (`files/demo/large_user.json`) generated by algorithm templates. No personal submission records, usernames, or session identifiers are published.

---

## Technical Limitations & Honest Real-World Considerations

1. **Synthetic Training Population vs. Live Contest Ratings**:
   Because there is no publicly accessible, multi-user LeetCode dataset with labeled submission event streams and historical contest ratings, the offline models are trained on a controlled synthetic population (600 users across 9 distinct archetypes). While the feature relationships and skill distributions are grounded in competitive programming dynamics, the predicted contest rating is an **offline benchmark proxy** rather than an official LeetCode rating.
2. **Cold-Start Fold-In Guard**:
   The closed-form ALS fold-in algorithm ($x_u = (Y^T C_u Y + \lambda I)^{-1} Y^T C_u p_u$) requires at least 3 problem interactions to produce a stable collaborative filtering vector. For brand-new users with 0–2 submissions, collaborative filtering is marked unavailable and the system seamlessly falls back to content-based topic similarity and structured weakness boosting.
3. **Zero Retraining on Online Serving**:
   To guarantee sub-second dashboard latency and prevent catastrophic overfitting on a single user's small sample, neither XGBoost nor ALS is retrained online during report generation. All single-user inference uses pre-trained offline models and algebraic fold-in on fixed item factor matrices.
4. **Canonical Question Universe**:
   The canonical catalogue contains 453 problems covering 20 major algorithmic topics. Real submissions outside this universe are aligned via problem title/slug matching or mapped to canonical topic tags.

---

## Future Improvements

- [ ] Support for problem solution code AST analysis to detect syntactic and time-complexity antipatterns.
- [ ] Integration with spaced-repetition schedules (e.g., SuperMemo SM-2) for periodic review of historically failed problems.
- [ ] Multi-platform ingestion support (Codeforces, HackerRank, AtCoder).
- [ ] Local fine-tuned LLM explanation generation for customized hint provision without giving away solutions.

---

## License

This project is licensed under the [MIT License](LICENSE).
