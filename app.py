"""
Flight Delay Prediction – Home Page
"""

import streamlit as st

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
)

# ── Title ──
st.title("✈️ Flight Delay Prediction")
st.caption(
    "Machine Learning Mini Project | **Manik Sharma** | Roll No: **CS3236** | TY Div B"
)

st.divider()

# ── Project info ──
st.header("📋 About This Project")

st.markdown("""
**Problem:** Predict *before take-off* whether a flight will arrive more than
15 minutes late at its destination.  This is a **binary classification** task
(delayed = 1, on-time = 0).

**Dataset:** The data comes from two public US government sources:

| Source | What it provides |
|--------|-----------------|
| **nycflights13** (Bureau of Transportation Statistics) | Every domestic flight departing the three New York airports (EWR, JFK, LGA) in 2013 |
| **Iowa Environmental Mesonet** | Hourly weather observations at each airport |

> **No cheating:** Only information that is known *before* departure is used as
> input.  The actual departure delay is **not** among the features, so the
> models cannot simply copy it.
""")

st.divider()

# ── Metric tiles ──
st.header("📊 Dataset at a Glance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Full dataset", "336,776 flights")
c2.metric("Used in this app", "12,000 flights")
c3.metric("Delayed flights", "23.7 %")
c4.metric("Algorithms", "6")

st.divider()

# ── Algorithms ──
st.header("🤖 The Six Algorithms")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Logistic Regression")
    st.markdown(
        "A linear model that estimates the probability of delay using a "
        "logistic (sigmoid) function. Fast and interpretable."
    )

    st.subheader("2. K-Nearest Neighbours (KNN)")
    st.markdown(
        "Classifies a flight by looking at the *k* most similar flights in the "
        "training set and taking a distance-weighted vote."
    )

with col2:
    st.subheader("3. Support Vector Machine (SVM)")
    st.markdown(
        "Finds the hyperplane that best separates delayed and on-time flights "
        "in a high-dimensional feature space (RBF kernel)."
    )

    st.subheader("4. Random Forest")
    st.markdown(
        "An ensemble of 200 decision trees, each trained on a random subset "
        "of the data. Robust and handles non-linear relationships well."
    )

with col3:
    st.subheader("5. AdaBoost")
    st.markdown(
        "Adaptive Boosting trains a sequence of weak learners (decision stumps), "
        "each focusing on the mistakes of the previous ones."
    )

    st.subheader("6. Ensemble (Voting)")
    st.markdown(
        "Combines Logistic Regression, KNN, Random Forest, and AdaBoost via "
        "soft voting – averaging their predicted probabilities."
    )

st.divider()

# ── Feature Selection ──
st.header("🔍 Feature Selection")

st.markdown("""
This app supports **optional feature selection** using `SelectKBest` from scikit-learn.
When enabled, the preprocessing pipeline applies a scoring function to all transformed
features (after one-hot encoding) and keeps only the top-*k* most informative ones.

| Method | How it scores features |
|--------|----------------------|
| **ANOVA F-test** (`f_classif`) | Measures the linear dependency between each feature and the target. Fast. |
| **Mutual Information** | Measures any (including non-linear) dependency between feature and target. Slower. |

Feature selection can improve model performance by removing noisy or redundant features,
and speeds up training.
""")

st.divider()

# ── Features ──
st.header("📝 Input Features (15)")

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    st.markdown("""
**Flight information**
- `month` – Month of departure (1–12)
- `dow` – Day of week (0 = Mon … 6 = Sun)
- `hour` – Scheduled departure hour (5–23)
- `carrier` – Airline code (e.g. UA, DL, AA)
- `origin` – Departure airport (EWR / JFK / LGA)
- `dest` – Destination airport code
- `distance` – Flight distance in miles
""")

with feat_col2:
    st.markdown("""
**Weather at departure**
- `temp` – Temperature (°F)
- `dewp` – Dew point (°F)
- `humid` – Relative humidity (%)
- `wind_dir` – Wind direction (degrees)
- `wind_speed` – Wind speed (mph)
- `precip` – Precipitation (inches)
- `pressure` – Sea-level pressure (mbar)
- `visib` – Visibility (miles)
""")

st.divider()

# ── How to use ──
st.header("🚀 How to Use This App")

st.markdown("""
1. **Train** – Go to the *Train Data* page.  Load the built-in dataset (or
   upload your own CSV), optionally enable feature selection, choose the
   train/test split, and click **Train models**.
2. **Test** – Go to the *Test Data* page.  Test on the held-out test set,
   upload a new CSV, or fill in a single flight's details to see live predictions
   from all six algorithms.
3. **Report** – Go to the *Algorithm Report* page to compare all six
   algorithms side by side with ROC curves, feature importance charts,
   confusion matrices, and plain-language explanations.
""")
