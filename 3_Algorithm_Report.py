"""
Page 3 – Algorithm Report
Side-by-side comparison of all six algorithms with ROC curves,
feature importance analysis, and per-algorithm tabs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    ALGORITHM_NAMES, bundle_exists, load_bundle,
    get_feature_importance, get_selectkbest_scores,
)

st.set_page_config(page_title="Algorithm Report", page_icon="📊", layout="wide")
st.title("📊 Algorithm Report")

# ── Guard ──
if not bundle_exists():
    st.warning("⚠️ No trained models found.  Please go to the **Train Data** page first.")
    st.stop()

bundle = load_bundle()
metrics_df = bundle["metrics_df"]
cm_dict = bundle["cm_dict"]
roc_dict = bundle["roc_dict"]
fs_info = bundle.get("feature_selection", {})

# =====================================================================
# Feature Selection info
# =====================================================================
if fs_info.get("enabled"):
    st.info(
        f"🔍 **Feature selection was enabled** during training: "
        f"top **{fs_info['k_best']}** features kept using **{fs_info['method']}**."
    )

# =====================================================================
# Overall comparison
# =====================================================================
st.header("Overall Comparison")

metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_max]

styled = metrics_df.style.apply(highlight_max, subset=metric_cols)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── ROC Curves ──
st.subheader("ROC Curves")

colors = {
    "Logistic Regression": "#3498db",
    "K-Nearest Neighbours": "#e67e22",
    "Support Vector Machine": "#9b59b6",
    "Random Forest": "#2ecc71",
    "AdaBoost": "#e74c3c",
    "Ensemble (Voting)": "#1abc9c",
}

fig_roc, ax_roc = plt.subplots(figsize=(8, 5.5))
for name in ALGORITHM_NAMES:
    if name in roc_dict:
        fpr, tpr, auc = roc_dict[name]
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})",
                    color=colors.get(name, None), lw=2)
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves – All Algorithms")
ax_roc.legend(loc="lower right", fontsize=7)
fig_roc.tight_layout()
st.pyplot(fig_roc)

# ── Metric explanations ──
st.subheader("Metric Glossary")
st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Accuracy** | Overall fraction of correct predictions (both classes). |
| **Precision** | Of flights predicted *delayed*, how many were actually delayed? (Low false alarms) |
| **Recall** | Of flights that were *actually delayed*, how many did the model catch? (Low missed delays) |
| **F1-Score** | Harmonic mean of Precision and Recall – a single number balancing both. |
| **ROC-AUC** | Area Under the ROC Curve – measures how well the model separates the two classes across all thresholds. 1.0 is perfect; 0.5 is random. |
""")

st.divider()

# =====================================================================
# Feature Importance
# =====================================================================
st.header("🔍 Feature Importance")

try:
    agg_df, raw_df = get_feature_importance(bundle)

    fi_col1, fi_col2 = st.columns(2)

    with fi_col1:
        st.markdown("**Original features (aggregated)**")
        fig_agg, ax_agg = plt.subplots(figsize=(5, 4))
        ax_agg.barh(
            agg_df["Feature"][::-1],
            agg_df["Importance"][::-1],
            color="#3498db",
        )
        ax_agg.set_xlabel("Importance")
        ax_agg.set_title("Feature Importance (Random Forest)")
        fig_agg.tight_layout()
        st.pyplot(fig_agg)

    with fi_col2:
        st.markdown("**Top 25 transformed features**")
        top = raw_df.head(25)
        fig_raw, ax_raw = plt.subplots(figsize=(5, 6))
        ax_raw.barh(
            top["Feature"][::-1].astype(str),
            top["Importance"][::-1],
            color="#e67e22",
        )
        ax_raw.set_xlabel("Importance")
        ax_raw.set_title("Top 25 Transformed Features (RF)")
        ax_raw.tick_params(axis="y", labelsize=7)
        fig_raw.tight_layout()
        st.pyplot(fig_raw)

    st.caption(
        "Aggregated importance sums the Random Forest feature importances of all "
        "one-hot encoded columns back to the original feature. Higher = more "
        "influential in splitting decisions."
    )

    # SelectKBest scores (if feature selection was used)
    skb_df = get_selectkbest_scores(bundle)
    if skb_df is not None:
        with st.expander("📋 SelectKBest feature scores", expanded=False):
            st.dataframe(skb_df, use_container_width=True, hide_index=True)
            n_sel = skb_df["Selected"].sum()
            st.markdown(f"**{n_sel}** features selected out of **{len(skb_df)}** total.")

except Exception as e:
    st.warning(f"Could not extract feature importance: {e}")

st.divider()

# =====================================================================
# Per-algorithm tabs
# =====================================================================
st.header("Per-Algorithm Details")

ALGO_EXPLANATIONS = {
    "Logistic Regression": """
**How it works:** Logistic Regression models the probability of the positive
class (delayed) as a sigmoid function of a linear combination of the input
features.  During training, it adjusts the weights of each feature to maximise
the likelihood of the observed labels.

**Settings used:**
- `C = 0.1` (inverse regularisation strength – smaller = more regularisation)
- `class_weight = "balanced"` (up-weights the minority "delayed" class)
- `max_iter = 1000`

**Strengths:** Fast, interpretable, works well when the decision boundary is
roughly linear.

**Weaknesses:** Cannot capture complex non-linear relationships without
feature engineering.
""",
    "K-Nearest Neighbours": """
**How it works:** KNN is a lazy learner – it stores the entire training set and
makes predictions by finding the *k* nearest data points (neighbours) to a new
observation.  The prediction is a weighted vote of the neighbours' labels, with
closer neighbours counting more.

**Settings used:**
- `n_neighbors = 5`
- `weights = "distance"` (closer neighbours have more influence)

**Strengths:** Simple, no training phase, adapts to any decision boundary shape.

**Weaknesses:** Slow at prediction time for large datasets; sensitive to
irrelevant features and the curse of dimensionality.
""",
    "Support Vector Machine": """
**How it works:** SVM finds the hyperplane (decision boundary) that maximises
the margin between the two classes.  Using the RBF kernel, it implicitly maps
the data into a higher-dimensional space so that a non-linear boundary in the
original space becomes linear in the new space.

**Settings used:**
- `kernel = "rbf"` (Radial Basis Function)
- `C = 1` (regularisation parameter)
- `class_weight = "balanced"`
- `probability = False` (intentionally off for speed)

**Strengths:** Effective in high-dimensional spaces, robust to overfitting
with proper regularisation.

**Weaknesses:** Training can be slow on large datasets; no probability
output by default.
""",
    "Random Forest": """
**How it works:** Random Forest builds an ensemble of many decision trees, each
trained on a random bootstrap sample of the data and a random subset of
features.  The final prediction is the majority vote of all trees.  This
"bagging + feature randomness" reduces overfitting and variance.

**Settings used:**
- `n_estimators = 200` (number of trees)
- `min_samples_leaf = 5` (minimum samples in a leaf node)
- `class_weight = "balanced"`
- `n_jobs = -1` (use all CPU cores)

**Strengths:** Handles non-linear relationships, robust to noise, provides
feature importance scores.

**Weaknesses:** Less interpretable than a single decision tree; can be
memory-intensive with many trees.
""",
    "AdaBoost": """
**How it works:** AdaBoost (Adaptive Boosting) is a boosting ensemble method
that trains a sequence of weak classifiers (typically decision stumps – single-
split trees).  After each round the algorithm increases the weight of
misclassified samples so the next weak learner focuses on the hard cases.
The final prediction is a weighted vote of all weak learners.

**Settings used:**
- `n_estimators = 100` (number of boosting rounds)
- `learning_rate = 0.1` (shrinks the contribution of each weak learner)
- Base estimator: `DecisionTreeClassifier(max_depth=1)` (default)

**Strengths:** Simple, effective at reducing bias, less prone to overfitting
than many other boosting methods when the learning rate is low.

**Weaknesses:** Sensitive to noisy data and outliers (misclassified noisy
points get very high weights); slower than a single decision tree.
""",
    "Ensemble (Voting)": """
**How it works:** The Voting Classifier combines predictions from multiple
diverse base models.  With **soft voting**, it averages the predicted
probabilities from each base model and picks the class with the highest
average probability.

**Base learners:**
- Logistic Regression (C = 0.1, balanced)
- K-Nearest Neighbours (k = 5, distance-weighted)
- Random Forest (200 trees, balanced)
- AdaBoost (100 rounds, lr = 0.1)

> SVM is excluded because it does not produce probability estimates
> (probability=True is too slow on this dataset).

**Settings used:**
- `voting = "soft"` (average probabilities rather than hard majority vote)

**Strengths:** Combines the strengths of different model families; often
more robust than any single model; reduces variance.

**Weaknesses:** Slower to train (trains 4 models internally); less
interpretable; the ensemble is only as good as its base learners.
""",
}

tabs = st.tabs(ALGORITHM_NAMES)

for tab, name in zip(tabs, ALGORITHM_NAMES):
    with tab:
        if name not in bundle["fitted"]:
            st.info(f"{name} was not trained in this bundle.")
            continue

        st.subheader(name)

        # Explanation
        st.markdown(ALGO_EXPLANATIONS.get(name, ""))

        # Metrics
        row_mask = metrics_df["Algorithm"] == name
        if not row_mask.any():
            continue
        row = metrics_df[row_mask].iloc[0]

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy", f"{row['Accuracy']:.2%}")
        mc2.metric("Precision", f"{row['Precision']:.2%}")
        mc3.metric("Recall", f"{row['Recall']:.2%}")
        mc4.metric("F1-Score", f"{row['F1-Score']:.2%}")
        mc5.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}")

        # Confusion matrix
        cm = cm_dict[name]
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
        im = ax_cm.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                           fontsize=14,
                           color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred On-time", "Pred Delayed"])
        ax_cm.set_yticklabels(["Actual On-time", "Actual Delayed"])
        ax_cm.set_title(f"Confusion Matrix – {name}")
        fig_cm.colorbar(im, ax=ax_cm, shrink=0.8)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

        # ROC curve for this algorithm
        if name in roc_dict:
            fpr, tpr, auc = roc_dict[name]
            fig_r, ax_r = plt.subplots(figsize=(5, 4))
            ax_r.plot(fpr, tpr, color=colors.get(name, "#3498db"), lw=2,
                      label=f"AUC = {auc:.4f}")
            ax_r.plot([0, 1], [0, 1], "k--", lw=1)
            ax_r.set_xlabel("False Positive Rate")
            ax_r.set_ylabel("True Positive Rate")
            ax_r.set_title(f"ROC Curve – {name}")
            ax_r.legend(loc="lower right")
            fig_r.tight_layout()
            st.pyplot(fig_r)

        # Interpretation
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
**Interpretation:**
- **{tp:,}** delayed flights correctly caught (True Positives)
- **{fn:,}** delayed flights missed (False Negatives)
- **{fp:,}** on-time flights incorrectly flagged (False Positives)
- **{tn:,}** on-time flights correctly identified (True Negatives)
""")
