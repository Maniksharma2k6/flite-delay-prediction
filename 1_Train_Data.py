"""
Page 1 – Train Data
Load (or upload) data, configure feature selection, train models, view results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    DATA_PATH, FEATURE_COLS, TARGET_COL, CATEGORICAL_COLS,
    ALGORITHM_NAMES, FS_METHODS,
    load_csv, validate_columns, train_all, save_bundle,
)

st.set_page_config(page_title="Train Data", page_icon="🏋️", layout="wide")
st.title("🏋️ Train Data")

# ── 1. Data source ──
st.subheader("1️⃣ Choose a dataset")

source = st.radio(
    "Data source",
    ["Use the built-in dataset (data/flights_sample.csv)", "Upload my own CSV file"],
    horizontal=True,
)

df = None

if source.startswith("Use"):
    try:
        df = load_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Built-in dataset not found at `{DATA_PATH}`.")

else:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        df = load_csv(uploaded)

if df is not None:
    missing = validate_columns(df, need_target=True)
    if missing:
        st.error(f"The uploaded CSV is missing these required columns: **{', '.join(missing)}**")
        st.stop()

    # Cast month/dow
    for col in ["month", "dow"]:
        df[col] = df[col].astype(str)

    delay_rate = df[TARGET_COL].mean()
    st.success(
        f"✅ Loaded **{len(df):,}** rows  ·  Delay rate **{delay_rate:.1%}**"
    )

    # ── Preview ──
    st.subheader("2️⃣ Data preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ── Charts ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Class distribution**")
        class_counts = df[TARGET_COL].value_counts().sort_index()
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        bars = ax1.bar(
            ["On time (0)", "Delayed (1)"],
            class_counts.values,
            color=["#2ecc71", "#e74c3c"],
        )
        for bar in bars:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{int(bar.get_height()):,}",
                ha="center", fontsize=10,
            )
        ax1.set_ylabel("Number of flights")
        ax1.set_title("Class Distribution")
        fig1.tight_layout()
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("**Delay rate by hour**")
        hourly = df.groupby("hour")[TARGET_COL].mean().sort_index()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(hourly.index.astype(int), hourly.values, marker="o", color="#3498db")
        ax2.set_xlabel("Scheduled departure hour")
        ax2.set_ylabel("Delay rate")
        ax2.set_title("Delay Rate by Hour")
        ax2.set_ylim(0, max(hourly.values) * 1.2)
        fig2.tight_layout()
        st.pyplot(fig2)

    # ── Feature Selection ──
    st.subheader("3️⃣ Feature Selection (optional)")

    use_fs = st.checkbox("Enable feature selection (SelectKBest)", value=False)

    k_best = None
    fs_method_name = list(FS_METHODS.keys())[0]

    if use_fs:
        fs_col1, fs_col2 = st.columns(2)
        with fs_col1:
            fs_method_name = st.selectbox(
                "Scoring method",
                list(FS_METHODS.keys()),
                help="ANOVA F-test is fast and works well for classification. "
                     "Mutual Information captures non-linear relationships but is slower.",
            )
        with fs_col2:
            k_best = st.slider(
                "Number of features to keep (k)",
                min_value=5, max_value=80, value=20, step=1,
                help="After one-hot encoding the 15 original features expand to ~140+ "
                     "transformed features. SelectKBest keeps the top-k most informative ones.",
            )
        st.info(
            f"🔍 **Feature selection enabled:** the top **{k_best}** transformed features "
            f"will be selected using **{fs_method_name}** before training each model."
        )

    # ── Training controls ──
    st.subheader("4️⃣ Train models")

    test_pct = st.slider(
        "Test set percentage", min_value=10, max_value=40, value=20, step=1
    )
    test_size = test_pct / 100.0

    st.markdown(
        "Training **6 algorithms**: Logistic Regression, KNN, SVM, "
        "Random Forest, AdaBoost, and Ensemble (Voting)."
    )

    if st.button("🚀 Train models", type="primary"):
        progress = st.progress(0, text="Starting…")

        def update(frac, msg):
            progress.progress(frac, text=msg)

        results = train_all(
            df,
            test_size=test_size,
            k_best=k_best,
            fs_method_name=fs_method_name,
            progress_cb=update,
        )
        save_bundle(results)

        st.session_state["train_results"] = results
        st.success("✅ All 6 models trained and saved to `models/bundle.joblib`")

    # ── Results (show after training) ──
    if "train_results" in st.session_state:
        results = st.session_state["train_results"]
        metrics_df = results["metrics_df"]

        st.subheader("5️⃣ Results")

        # Highlight best per metric column
        st.markdown("**Metrics on held-out test set**")
        metric_cols_to_highlight = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

        def highlight_max(s):
            is_max = s == s.max()
            return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_max]

        styled = metrics_df.style.apply(
            highlight_max, subset=metric_cols_to_highlight
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Bar chart – Precision / Recall / F1
        st.markdown("**Precision / Recall / F1-Score comparison**")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        x = np.arange(len(metrics_df))
        width = 0.25
        ax3.bar(x - width, metrics_df["Precision"], width, label="Precision", color="#3498db")
        ax3.bar(x,         metrics_df["Recall"],    width, label="Recall",    color="#e74c3c")
        ax3.bar(x + width, metrics_df["F1-Score"],  width, label="F1-Score",  color="#2ecc71")
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_df["Algorithm"], rotation=20, ha="right", fontsize=7)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Score")
        ax3.legend()
        ax3.set_title("Precision / Recall / F1-Score")
        fig3.tight_layout()
        st.pyplot(fig3)

        # Confusion matrix selector
        st.markdown("**Confusion Matrix**")
        selected_algo = st.selectbox(
            "Select algorithm", ALGORITHM_NAMES, key="cm_algo"
        )
        cm = results["cm_dict"][selected_algo]

        fig4, ax4 = plt.subplots(figsize=(4, 3.5))
        im = ax4.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["Pred On-time", "Pred Delayed"])
        ax4.set_yticklabels(["Actual On-time", "Actual Delayed"])
        ax4.set_title(f"Confusion Matrix – {selected_algo}")
        fig4.colorbar(im, ax=ax4, shrink=0.8)
        fig4.tight_layout()
        st.pyplot(fig4)

        st.info(
            "💡 **Note:** Always predicting *on time* would give ~76% accuracy "
            "but would catch **zero** delays.  That's why **Recall** and "
            "**F1-Score** matter more than raw accuracy for this problem."
        )

else:
    if source.startswith("Upload"):
        st.info("👆 Upload a CSV file to get started.")
