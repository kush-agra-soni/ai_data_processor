# ui/streamlit_app.py

import json
import tempfile
from typing import Dict, Any

import streamlit as st
import pandas as pd
import yaml

from ai.planner import get_planner
from ai.executor import get_executor, ExecutorLogger
from core._1_loader import load_file


# ==========================================================
# Streamlit setup
# ==========================================================
st.set_page_config(
    page_title="AI Data Cleaning Pipeline",
    layout="wide"
)

st.title("🧠 AI-Driven Data Cleaning Pipeline")
st.caption("Profiler → Planner → Executor → Validation")


# ==========================================================
# Helpers
# ==========================================================
def load_config(path: str = "configs/ai_config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class StreamlitLogger(ExecutorLogger):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.logs = []

    def log(self, message: str):
        self.logs.append(message)
        self.placeholder.code("\n".join(self.logs))


# ==========================================================
# Sidebar controls
# ==========================================================
st.sidebar.header("⚙️ Settings")

config = load_config()

ai_mode = st.sidebar.selectbox(
    "AI Mode",
    ["autonomous", "advisory"],
    index=0 if config["ai"]["mode"] == "autonomous" else 1
)

strict_mode = st.sidebar.checkbox(
    "Strict execution mode",
    value=config["ai"]["strict_mode"]
)

target_column = st.sidebar.text_input(
    "Target column (optional)",
    help="Required for feature selection / resampling"
)

show_profile = st.sidebar.checkbox("Show dataset profile", value=True)
show_plan = st.sidebar.checkbox("Show AI plan JSON", value=True)


# ==========================================================
# File upload
# ==========================================================
st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV / JSON / XLSX file",
    type=["csv", "json", "xlsx"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    df = load_file(file_path)
    st.success(f"Loaded dataset with shape {df.shape}")

    st.subheader("🔍 Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # ======================================================
    # AI Planning
    # ======================================================
    st.header("🧠 AI Planning")

    planner = get_planner()

    with st.spinner("Profiling dataset and generating plan..."):
        decision = planner.plan(df)

    if decision["confidence"] < config["ai"]["confidence_threshold"]:
        st.warning(
            f"Low AI confidence: {decision['confidence']:.2f} "
            f"(threshold {config['ai']['confidence_threshold']})"
        )

    if show_plan:
        st.subheader("📜 AI Plan (JSON)")
        st.json(decision)

    # ======================================================
    # Human approval (if advisory)
    # ======================================================
    allow_execution = True

    if ai_mode == "advisory":
        st.warning("Advisory mode enabled — manual approval required")
        allow_execution = st.checkbox("✅ Approve AI plan for execution")

    # ======================================================
    # Execution
    # ======================================================
    if allow_execution and st.button("🚀 Run Pipeline"):
        st.header("⚙️ Execution Logs")

        log_placeholder = st.empty()
        logger = StreamlitLogger(log_placeholder)

        executor = get_executor(
            logger=logger,
            strict_mode=strict_mode
        )

        try:
            result = executor.execute(
                df=df,
                decision=decision,
                target_column=target_column or None
            )

            st.success("Pipeline execution completed")

            # ==================================================
            # Results
            # ==================================================
            st.header("✅ Results")

            if result.get("data") is not None:
                st.subheader("Cleaned Dataset")
                st.dataframe(result["data"].head(50), use_container_width=True)

            if result.get("X") is not None:
                st.subheader("Feature Matrix (X)")
                st.dataframe(result["X"].head(50), use_container_width=True)

            if result.get("reports"):
                st.subheader("Validation Report")
                for report in result["reports"]:
                    st.json(report)

            if result.get("artifacts"):
                st.subheader("Artifacts & Metadata")
                st.json(result["artifacts"])

        except Exception as e:
            st.error(f"Execution failed: {e}")

else:
    st.info("Upload a dataset to begin.")
