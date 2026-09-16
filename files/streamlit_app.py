from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Ensure predictable module resolution across project subdirectories
FILES_DIR = Path(__file__).parent
ROOT_DIR = FILES_DIR.parent
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

import streamlit as st

import config
from data_processing import load_leetcode_history_export, load_leetcode_history_records
from main import LeetCodeMentor


st.set_page_config(page_title="LeetCode Mentor", layout="wide")

st.title("LeetCode Mentor — Local Report Generator")

st.markdown(
    """Use local sync data from the Chrome extension or upload a LeetCode export JSON file.
    The app processes your submission history, builds user features, analyzes weak subtopics, and provides tailored recommendations.
    """
)

if not config.RATING_MODEL_PATH.exists() or not config.ALS_MODEL_PATH.exists():
    st.warning(
        "⚠️ **Offline model artifacts not detected.** "
        "Run `python training/train_models.py` to pre-train the XGBoost contest predictor and ALS recommender."
    )

SYNC_STORE_PATH = Path(__file__).parent / "sync_store.json"
DEMO_STORE_PATH = Path(__file__).parent / "demo" / "large_user.json"

# Load genuine synced accounts from sync_store.json
synced_accounts = {}
if SYNC_STORE_PATH.exists():
    try:
        raw_store = json.loads(SYNC_STORE_PATH.read_text(encoding="utf-8"))
        for username, account in raw_store.items():
            if isinstance(account, dict) and isinstance(account.get("submissions"), list):
                # Ensure test/demo accounts are ignored in production sync selection
                is_demo = account.get("is_demo_data", False)
                if not is_demo and username != "large_user":
                    synced_accounts[username] = account
    except Exception:
        synced_accounts = {}

mode = st.radio(
    "Data Source",
    [
        "My LeetCode Data (Extension Sync)",
        "Upload LeetCode Export (JSON/CSV/TSV)",
        "Demo Dataset (Development / Testing Only)"
    ]
)

export_path: Path | None = None
selected_records: list[dict] | None = None
data_provenance: dict | None = None
can_generate: bool = False

if mode == "My LeetCode Data (Extension Sync)":
    if synced_accounts:
        username = st.selectbox("Choose synced account", list(synced_accounts.keys()))
        account_data = synced_accounts.get(username, {})
        selected_records = account_data.get("submissions", [])
        if selected_records:
            can_generate = True
            data_provenance = {
                "data_source": "leetcode_extension",
                "is_demo_data": False,
                "submission_count": len(selected_records),
                "username": username,
                "last_synced": account_data.get("last_synced", "Unknown")
            }
            st.success(f"Loaded {len(selected_records)} genuine synced submissions for user '{username}'.")
        else:
            st.warning(f"No submission records found for user '{username}'.")
    else:
        st.error(
            "No valid synchronized LeetCode data is available.\n\n"
            "The extension synchronization has not completed or failed (e.g. HTTP 403).\n"
            "Please log into LeetCode in your browser and click 'Sync with AI Mentor' in the Chrome extension, or upload an export file."
        )

elif mode == "Upload LeetCode Export (JSON/CSV/TSV)":
    uploaded = st.file_uploader("Choose export JSON/CSV/TSV", type=["json", "csv", "tsv"], accept_multiple_files=False)
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(uploaded.getvalue())
        tmp.flush()
        tmp.close()
        export_path = Path(tmp.name)
        can_generate = True
        data_provenance = {
            "data_source": "json_import",
            "is_demo_data": False,
            "filename": uploaded.name
        }
        st.success(f"Loaded uploaded export file: {uploaded.name}")
    else:
        st.info("Please select and upload your exported LeetCode history file.")

else:  # Demo Dataset (Development / Testing Only)
    st.warning("⚠️ **DEMO MODE ENABLED**: Using synthetic/demo test fixture for development testing.")
    if DEMO_STORE_PATH.exists():
        try:
            demo_data = json.loads(DEMO_STORE_PATH.read_text(encoding="utf-8"))
            selected_records = demo_data.get("submissions", [])
            if selected_records:
                can_generate = True
                data_provenance = {
                    "data_source": "demo",
                    "is_demo_data": True,
                    "submission_count": len(selected_records),
                    "username": demo_data.get("username", "large_user")
                }
                st.info(f"Loaded {len(selected_records)} synthetic submissions from demo fixture ('large_user').")
            else:
                st.error("Demo fixture contains no submission records.")
        except Exception as e:
            st.error(f"Failed to load demo dataset: {e}")
    else:
        st.error(f"Demo fixture file '{DEMO_STORE_PATH}' not found.")

st.markdown("---")

if st.button("Generate Report", disabled=not can_generate):
    if not can_generate:
        st.error("Report generation is blocked: No valid real dataset is selected or available.")
        st.stop()

    try:
        if mode == "My LeetCode Data (Extension Sync)" or mode == "Demo Dataset (Development / Testing Only)":
            if not selected_records:
                st.error("No submission records available to generate report.")
                st.stop()
            st.info("Loading submission records and running analysis pipeline...")
            users_df, questions_df, submissions_df = load_leetcode_history_records(selected_records)

        else:
            if export_path is None:
                st.error("Please upload a file first.")
                st.stop()
            st.info("Loading export file and running analysis pipeline...")
            users_df, questions_df, submissions_df = load_leetcode_history_export(str(export_path))

        mentor = LeetCodeMentor(users_df, questions_df, submissions_df)
        user_choice = int(mentor.users_df["user_id"].iloc[0])
        report = mentor.generate_report(user_choice, provenance=data_provenance)
        st.session_state["report"] = report
        st.success("Mentor report successfully generated!")
    except Exception as e:
        import traceback
        st.error(f"Error running pipeline: {e}")
        st.code(traceback.format_exc())

if "report" in st.session_state:
    report = st.session_state["report"]

    st.markdown("---")
    st.header("Mentor Performance & Diagnostic Summary")

    prov = report.get("data_provenance", {})
    st.subheader("Data Provenance & Source Metadata")
    st.json(prov)

    diag = report.get("diagnostics", {})
    rating_val = report.get("predicted_contest_rating")
    rating_display = f"{rating_val:.1f}" if rating_val is not None else "Rating prediction unavailable"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Submissions", diag.get("total_submissions", 0))
    col2.metric("Solved Submissions", diag.get("solved_submissions", 0))
    col3.metric("Failed Submissions", diag.get("failed_submissions", 0))
    col4.metric(
        "Benchmark Rating",
        rating_display,
        help="Estimated via offline-trained XGBoost pipeline. Represents synthetic benchmark proxy, not official LeetCode rating."
    )

    st.subheader("Data Quality Diagnostics")
    st.json(diag)

    # --------------------------------------------------------------------------
    # Structured Topic Weakness Analysis
    # --------------------------------------------------------------------------
    st.subheader("Topic Weakness Analysis")
    topic_profile = report.get("topic_profile", [])
    if topic_profile:
        import pandas as pd
        df_display = pd.DataFrame(topic_profile)[
            ["topic", "attempts", "success_rate_pct", "exposure_level", "dominant_difficulty", "weakness_level"]
        ].rename(columns={
            "topic": "Topic",
            "attempts": "Attempts",
            "success_rate_pct": "Success",
            "exposure_level": "Exposure",
            "dominant_difficulty": "Difficulty",
            "weakness_level": "Risk",
        })
        st.table(df_display)

        # Why this is weak breakdown
        weak_topics_explanations = [t for t in topic_profile if t.get("why_weak")]
        if weak_topics_explanations:
            st.markdown("### Why Weakest Topics are Flagged")
            for t in weak_topics_explanations:
                st.markdown(f"#### **{t.get('topic')}** (`{t.get('weakness_level')}`)")
                st.markdown("**Why this is weak:**")
                for bullet in t.get("why_weak", []):
                    st.markdown(f"- {bullet}")
    else:
        st.info("Insufficient submission data to compute detailed topic weakness profile.")

    # Expandable Secondary NLP Subpattern Analysis
    with st.expander("Secondary NLP Subpattern Analysis & Raw Clusters (Technical / Debug)"):
        weak_info = report.get("weak_subtopics", {})
        if "note" in weak_info:
            st.info(weak_info["note"])
        clusters = weak_info.get("clusters", [])
        if not clusters:
            st.write("No distinct weak topic clusters identified.")
        else:
            for c in clusters:
                st.markdown(f"**Cluster {c.get('cluster_id')}** — Size: {c.get('size')} failure(s)")
                st.markdown(f"- **Keywords / Weak Topics**: {', '.join(c.get('keywords', []))}")
                st.markdown(f"- **Sample Problems**: {', '.join(c.get('sample_titles', []))}")

    # --------------------------------------------------------------------------
    # Top Recommended Questions
    # --------------------------------------------------------------------------
    st.subheader("Top Recommended Questions")
    recs = report.get("recommended_questions", [])
    if not recs:
        st.write("No recommendations generated.")
    else:
        for i, r in enumerate(recs, start=1):
            score = r.get("recommendation_score", 0.0)
            reason = r.get("reason", "Recommended practice")
            st.markdown(
                f"**{i}. Q{r.get('question_id')} | {r.get('title')}** `[{r.get('difficulty')}]`  \n"
                f"- **Tags**: {', '.join(r.get('topic_tags', []))}  \n"
                f"- **Score**: `{score:.3f}` | **Reason**: {reason}"
            )

    st.subheader("Contest Rating Model Status")
    eval_info = report.get("model_evaluation", {})
    if eval_info.get("available"):
        st.write(
            f"**Offline Test Metrics (Untouched Users)** — RMSE: `{eval_info.get('rmse')}` | "
            f"MAE: `{eval_info.get('mae')}` | R²: `{eval_info.get('r2')}`"
        )
        if eval_info.get("benchmark_note"):
            st.caption(f"ℹ️ {eval_info['benchmark_note']}")
    else:
        st.info(eval_info.get("note", "Model evaluation unavailable."))

    with st.expander("View Full Raw Report JSON"):
        st.json(report)

