# LeetCode Mentor: ML-Powered Algorithmic Performance Profiler & Recommendation System

> An end-to-end pair-programming mentor that ingests raw LeetCode submission history to build structured topic weakness profiles, extract semantic failure subpatterns with transformers, and recommend targeted practice problems via hybrid filtering.

---

## Overview

Practicing algorithmic coding problems on platforms like LeetCode often suffers from an unstructured "spray-and-pray" approach: candidates repeatedly solve comfortable problems while struggling to identify specific algorithmic weaknesses, time-decayed skill degradation, or recurring problem-pattern failure modes.

**LeetCode Mentor** solves this by providing an automated, privacy-first profiling and recommendation pipeline. It runs locally, ingests authentic submission logs directly from the browser via a lightweight Chrome extension or JSON/CSV exports, evaluates weakness across canonical algorithmic topics, uncovers semantic failure subpatterns using NLP clustering, and serves personalized recommendations through an interactive Streamlit dashboard.

---

## Key Features

- **Automated Local Sync**: Ingests LeetCode submission history incrementally or via full history rebuild using a local-only Chrome extension (Manifest V3) and FastAPI sync server.
- **Structured Topic Weakness Profiling**: Evaluates user proficiency across 40+ canonical algorithmic taxonomy tags (e.g., Dynamic Programming, Graph Traversal, Binary Search, Trees) using empirical success rates, problem exposure levels, and difficulty weighting.
- **Time-Decayed Recency Momentum**: Computes exponential time-decayed accuracy ($w_i = 0.5^{\Delta t / \tau}$, $\tau = 21\text{ days}$) to reward current problem-solving capability over historical performance.
- **Secondary NLP Subpattern Discovery**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate dense semantic embeddings of failed problem descriptions, clusters them with K-Means (optimal $k$ selected via silhouette score), and extracts human-interpretable keyword signatures using class-based TF-IDF (c-TF-IDF).
- **Hybrid Recommendation Engine**: Blends implicit-feedback Matrix Factorization trained via Alternating Least Squares (ALS) with tag-based Content Filtering and weakness profile boosting, enforcing topic diversity and excluding solved problems.
- **Contest Rating Estimation**: Uses XGBoost regression over behavioral feature vectors with explicit diagnostics and heuristic fallback when running on single-user datasets.
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
│   - Exposes status and export upload endpoints                   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             ML Analytics & Recommendation Pipeline               │
│                                                                  │
│  ┌─────────────────────────┐     ┌────────────────────────────┐  │
│  │ Feature Engineering     │     │ Weak Topic NLP Clustering  │  │
│  │ - Accuracy ratios       │     │ - all-MiniLM-L6-v2 dense   │  │
│  │ - Recency momentum      │───► │   embeddings of failures   │  │
│  │ - Structured profiling  │     │ - Silhouette score K-Means │  │
│  │ - Canonical taxonomy    │     │ - c-TF-IDF cluster tags    │  │
│  └───────────┬─────────────┘     └────────────────────────────┘  │
│              │                                                   │
│              ├───────────────────► ┌──────────────────────────┐  │
│              │                     │ Hybrid Recommender       │  │
│              │                     │ - Implicit ALS MF        │  │
│              │                     │ - Tag Cosine Similarity  │  │
│              │                     │ - Weakness Profile Boost │  │
│              │                     │ - Topic Diversity Guard  │  │
│              │                     └──────────────────────────┘  │
│              │                                                   │
│              └───────────────────► ┌──────────────────────────┐  │
│                                    │ Contest Rating Predictor │  │
│                                    │ - XGBoost Regression     │  │
│                                    │ - Multi-user evaluation  │  │
│                                    │ - Single-user fallback   │  │
│                                    └──────────────────────────┘  │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│             Streamlit Dashboard (files/streamlit_app.py)         │
│   - Visual diagnostic summary & data provenance                  │
│   - Tabular topic weakness matrix with actionable reasons        │
│   - Top-5 profile-guided recommendations with evidence context   │
│   - Expandable secondary NLP subpattern clusters                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Machine Learning & Algorithmic Methodology

### 1. Feature Engineering & Recency Momentum
The feature engineering pipeline (`files/data_processing.py`) aggregates raw submission event streams into user-level vectors without per-row loops:
- **Accuracy by Difficulty**: Ratios of accepted attempts across `Easy`, `Medium`, and `Hard` problems.
- **Difficulty Distribution**: Proportion of total practice effort spent on each difficulty band.
- **Exponential Recency Momentum**:
  $$\text{momentum} = \frac{\sum_i w_i \cdot \mathbb{I}(\text{status}_i = \text{Accepted})}{\sum_i w_i}, \quad w_i = 0.5^{\frac{\text{days\_ago}_i}{\tau}}$$
  where $\tau = 21.0\text{ days}$. This weights recent problem-solving success significantly higher than older historical attempts.

### 2. Structured Topic Weakness Profiling
- Maps raw tags to 40+ canonical LeetCode algorithmic topics.
- Calculates an empirical **Risk Score** for each topic:
  $$\text{Risk Score} = (1.0 - \text{Success Rate}) \times \text{Difficulty Weight} \times \text{Exposure Factor}$$
- Classifies topics into risk tiers: `Critical`, `Weak`, `Moderate`, `Neutral`, and `Strong`.
- Filters out non-algorithmic lexical noise (e.g., `"Bananas"`, `"Valid"`, `"Insert"`) from primary topic profiles.

### 3. Weak Topics NLP Clustering
For failed submissions (Wrong Answer, Time Limit Exceeded, Runtime Error, Compile Error):
1. **Semantic Embeddings**: Encodes problem descriptions into 384-dimensional dense vectors using `sentence-transformers` (`all-MiniLM-L6-v2`) with unit-norm normalization.
2. **Dynamic Model Selection**: Sweeps cluster count $k \in [3, 10]$ and selects the optimal $k$ maximizing the **Silhouette Score**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
3. **c-TF-IDF Topic Labeling**: Combines all descriptions in cluster $c$ into a composite document, computing class-based TF-IDF to assign descriptive keywords (e.g., `["Graph", "Breadth-First Search", "Shortest Path"]`) to the cluster.

### 4. Hybrid Recommendation Engine
- **Implicit ALS Matrix Factorization**: Solves closed-form Alternating Least Squares with confidence scaling $C_{ui} = 1 + \alpha R_{ui}$ ($\alpha = 15.0$, $\lambda = 20.0$, $d = 24$) to capture latent user-problem affinities.
- **Content-Based Similarity**: Computes cosine similarity over canonical topic tag indicator matrices to handle sparse/cold-start items.
- **Blended Score**:
  $$\text{Score} = w_{\text{cf}} \cdot \text{ALS}_{\text{norm}} + w_{\text{content}} \cdot \text{Content}_{\text{norm}} + 0.85 \cdot \text{RiskBoost}$$
- **Post-Processing Rules**:
  - Strictly excludes problems already solved by the user.
  - Enforces topic diversity (maximum 2 recommendations per primary topic tag).
  - Generates transparent, evidence-grounded reasons explaining why each problem was recommended.

### 5. Contest Rating Regression
- Trains an `XGBoostRegressor` on user behavioral feature matrices with StandardScaler preprocessing in an `sklearn.pipeline.Pipeline`.
- Evaluated via 5-fold cross-validation or held-out test split (reporting RMSE, MAE, and $R^2$).
- **Single-User Graceful Degradation**: On local single-user datasets where multi-user training samples are unavailable, the system transparently indicates that supervised training requires a multi-user distribution and provides an estimated heuristic rating.

---

## Tech Stack

| Domain | Technologies & Libraries |
|---|---|
| **Core ML & NLP** | Python 3.10+, NumPy, Pandas, Scipy, Scikit-Learn, PyTorch, Sentence-Transformers, XGBoost |
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
├── files/                   # Backend & ML Source Code
│   ├── api_server.py        # Local FastAPI sync server
│   ├── config.py            # Central pipeline constants & hyperparameters
│   ├── data_processing.py   # Synthetic generator & vectorized feature engineering
│   ├── export_helper.py     # CLI utilities for LeetCode export parsing
│   ├── leetcode_fetcher.py  # Session-based fetch helpers
│   ├── main.py              # LeetCodeMentor facade & CLI orchestrator
│   ├── nlp_cluster.py       # SentenceTransformers + K-Means + c-TF-IDF pipeline
│   ├── predictor.py         # XGBoost contest rating regression model
│   ├── recommender.py       # Implicit ALS + Content-Based hybrid recommender
│   ├── schemas.py           # Typed schema contracts for Question/User/Submission
│   ├── streamlit_app.py     # Interactive mentor dashboard UI
│   └── demo/
│       └── large_user.json  # Synthetic demo dataset for testing/evaluation
└── tests/                   # Unit & Integration Test Suite
    ├── __init__.py
    ├── test_api_server.py       # FastAPI sync & status endpoint tests
    ├── test_data_processing.py  # Synthetic data & feature engineering tests
    ├── test_end_to_end.py       # Full pipeline facade integration tests
    ├── test_nlp_cluster.py      # Embedding & clustering tests
    ├── test_predictor.py        # Rating prediction & fallback tests
    └── test_recommender.py      # ALS & hybrid recommender tests
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
*All 17 unit and integration tests should pass.*

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
2. **Click "Generate Report"**: The application executes the feature extraction, topic weakness analysis, NLP failure clustering, and hybrid recommendation pipeline.
3. **Inspect the Results**:
   - **Diagnostic Metrics**: Solved vs. failed submission counts, submission frequency, and contest rating estimate.
   - **Topic Weakness Matrix**: Interactive table showing topic attempts, success rate %, exposure level, and flagged weakness levels (`Critical`, `Weak`, `Moderate`).
   - **Actionable Explanations**: "Why Weakest Topics are Flagged" explains why a topic was flagged (e.g., high failure rate on Medium problems despite multiple attempts).
   - **Top-5 Recommendations**: Curated practice problems with score breakdown, tags, difficulty, and an explicit reason grounded in your weakness profile.
   - **Secondary NLP Subpatterns**: Expandable view showing semantically clustered problem failure patterns with extracted keyword signatures.

---

## Privacy & Security Architecture

This repository is designed with a strict **local-first privacy model**:

- **No Remote Credential Transmission**: The Chrome extension operates inside the user's active browser session on `leetcode.com` and communicates **exclusively** with `http://127.0.0.1:8000`. No cookies, tokens, or submission data are ever transmitted to third-party endpoints.
- **Local Persistence Only**: Synced submission records are written to a local file (`files/sync_store.json`), which is explicitly ignored by `.gitignore` and never committed to version control.
- **Sanitized Demo Data**: The repository includes only synthetic fixtures (`files/demo/large_user.json`) generated by algorithm templates. No personal submission records, usernames, or session identifiers are published.

---

## Technical Limitations & Real-World Considerations

- **Single-User Rating Regression**: Supervised rating models (like XGBoost) require a large multi-user population with diverse historical contest ratings to train unbiased parameters. On a local single-user machine, supervised training metrics are disabled, and the system transparently utilizes a heuristic rating approximation based on difficulty-weighted accuracy.
- **Topic Tag Availability**: Some raw LeetCode REST endpoints omit topic tag arrays for older submissions. The pipeline incorporates fallback pattern-matching against a canonical 40+ topic taxonomy to infer tags when explicit arrays are absent.
- **Recommendation Universe**: Recommendations are generated from the available question catalog in the pipeline (or synthetic question pool). In an enterprise deployment, this can be connected to a database containing the complete LeetCode question bank.

---

## Future Improvements

- [ ] Support for problem solution code AST analysis to detect syntactic and time-complexity antipatterns.
- [ ] Integration with spaced-repetition schedules (e.g., SuperMemo SM-2) for periodic review of historically failed problems.
- [ ] Multi-platform ingestion support (Codeforces, HackerRank, AtCoder).
- [ ] Local fine-tuned LLM explanation generation for customized hint provision without giving away solutions.

---

## License

This project is licensed under the [MIT License](LICENSE).
