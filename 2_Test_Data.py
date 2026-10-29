"""
Page 2 – Test Data
Test models on CSV or a single flight.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils import (
    FEATURE_COLS, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL,
    ALGORITHM_NAMES, HOLDOUT_PATH,
    load_csv, validate_columns, bundle_exists, load_bundle,
    predict_batch, predict_single, compute_metrics,
)

st.set_page_config(page_title="Test Data", page_icon="🧪", layout="wide")
st.title("🧪 Test Data")

# ── Guard: must train first ──
if not bundle_exists():
    st.warning("⚠️ No trained models found.  Please go to the **Train Data** page first.")
    st.stop()

bundle = load_bundle()

# ── Tabs ──
tab_csv, tab_single = st.tabs(["📁 Test a CSV file", "✈️ Test a single flight"])

# =====================================================================
# TAB 1 – Test a CSV
# =====================================================================
with tab_csv:
    csv_source = st.radio(
        "Test data source",
        ["Use the held-out test set", "Upload my own test CSV"],
        horizontal=True,
        key="test_csv_source",
    )

    test_df = None

    if csv_source.startswith("Use"):
        try:
            test_df = load_csv(HOLDOUT_PATH)
        except FileNotFoundError:
            st.error("Held-out test set not found. Please re-train the models.")
    else:
        uploaded = st.file_uploader("Upload test CSV", type=["csv"], key="test_upload")
        if uploaded is not None:
            test_df = load_csv(uploaded)

    if test_df is not None:
        # Validate features
        missing_feats = validate_columns(test_df, need_target=False)
        if missing_feats:
            st.error(f"Missing feature columns: **{', '.join(missing_feats)}**")
            st.stop()

        has_target = TARGET_COL in test_df.columns

        algo_choice = st.selectbox("Choose algorithm", ALGORITHM_NAMES, key="test_algo")
        pipe = bundle["fitted"][algo_choice]

        X_test = test_df[FEATURE_COLS].copy()
        for col in ["month", "dow"]:
            X_test[col] = X_test[col].astype(str)

        preds, scores = predict_batch(pipe, X_test)

        n_delayed = int((preds == 1).sum())
        n_ontime = int((preds == 0).sum())

        c1, c2 = st.columns(2)
        c1.metric("Predicted On time ✅", f"{n_ontime:,}")
        c2.metric("Predicted Delayed ❌", f"{n_delayed:,}")

        # Build results table
        results_df = test_df.copy()
        results_df["Prediction"] = np.where(preds == 1, "Delayed", "On time")

        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            results_df["Delay Probability"] = np.round(scores, 4)
        else:
            results_df["Decision Score"] = np.round(scores, 4)

        if has_target:
            y_true = test_df[TARGET_COL].values
            results_df["Correct"] = np.where(preds == y_true, "✅ Yes", "❌ No")

            metrics = compute_metrics(y_true, preds, scores)

            st.markdown("**Performance metrics**")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
            mc2.metric("Precision", f"{metrics['Precision']:.2%}")
            mc3.metric("Recall", f"{metrics['Recall']:.2%}")
            mc4.metric("F1-Score", f"{metrics['F1-Score']:.2%}")
            mc5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

            missed_delays = int(((y_true == 1) & (preds == 0)).sum())
            false_alarms = int(((y_true == 0) & (preds == 1)).sum())
            st.markdown(
                f"⚠️ **Missed delays** (predicted on-time but actually delayed): **{missed_delays}**  ·  "
                f"**False alarms** (predicted delayed but actually on-time): **{false_alarms}**"
            )

        st.markdown("**Predictions table**")
        st.dataframe(results_df.head(200), use_container_width=True)

        csv_download = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download all predictions as CSV",
            csv_download,
            "predictions.csv",
            "text/csv",
        )

# =====================================================================
# TAB 2 – Test a single flight
# =====================================================================
with tab_single:
    st.markdown("Fill in the flight and weather details below, then click **Predict**.")

    medians = bundle["median_defaults"]
    cats = bundle["cat_lists"]

    # Map month numbers to names for display
    MONTH_MAP = {
        "1": "January", "2": "February", "3": "March", "4": "April",
        "5": "May", "6": "June", "7": "July", "8": "August",
        "9": "September", "10": "October", "11": "November", "12": "December",
    }
    DOW_MAP = {
        "0": "Monday", "1": "Tuesday", "2": "Wednesday", "3": "Thursday",
        "4": "Friday", "5": "Saturday", "6": "Sunday",
    }

    # Reverse maps
    MONTH_REV = {v: k for k, v in MONTH_MAP.items()}
    DOW_REV = {v: k for k, v in DOW_MAP.items()}

    month_names = [MONTH_MAP.get(m, m) for m in cats.get("month", list(MONTH_MAP.keys()))]
    dow_names = [DOW_MAP.get(d, d) for d in cats.get("dow", list(DOW_MAP.keys()))]

    with st.form("single_flight"):
        st.markdown("#### ✈️ Flight Information")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            month_name = st.selectbox("Month", month_names)
            carrier = st.selectbox("Airline", cats.get("carrier", []))
            distance = st.number_input("Distance (miles)", value=int(medians.get("distance", 1000)), min_value=1)

        with fc2:
            dow_name = st.selectbox("Day of week", dow_names)
            origin = st.selectbox("Origin", cats.get("origin", ["EWR", "JFK", "LGA"]))

        with fc3:
            hour = st.slider("Departure hour", min_value=5, max_value=23, value=12)
            dest = st.selectbox("Destination", cats.get("dest", []))

        st.markdown("#### 🌤️ Weather at Departure")
        wc1, wc2, wc3, wc4 = st.columns(4)

        with wc1:
            temp = st.number_input("Temperature (°F)", value=round(medians.get("temp", 55.0), 1))
            dewp = st.number_input("Dew Point (°F)", value=round(medians.get("dewp", 40.0), 1))

        with wc2:
            humid = st.number_input("Humidity (%)", value=round(medians.get("humid", 55.0), 1))
            pressure = st.number_input("Pressure (mbar)", value=round(medians.get("pressure", 1017.0), 1))

        with wc3:
            wind_speed = st.number_input("Wind Speed (mph)", value=round(medians.get("wind_speed", 10.0), 1), min_value=0.0)
            wind_dir = st.number_input("Wind Direction (°)", value=round(medians.get("wind_dir", 240.0), 1), min_value=0.0, max_value=360.0)

        with wc4:
            precip = st.number_input("Precipitation (in)", value=round(medians.get("precip", 0.0), 2), min_value=0.0, format="%.2f")
            visib = st.number_input("Visibility (mi)", value=round(medians.get("visib", 10.0), 1), min_value=0.0)

        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        input_dict = {
            "month": MONTH_REV.get(month_name, "1"),
            "dow": DOW_REV.get(dow_name, "0"),
            "hour": hour,
            "carrier": carrier,
            "origin": origin,
            "dest": dest,
            "distance": distance,
            "temp": temp,
            "dewp": dewp,
            "humid": humid,
            "wind_dir": wind_dir,
            "wind_speed": wind_speed,
            "precip": precip,
            "pressure": pressure,
            "visib": visib,
        }

        results = predict_single(bundle, input_dict)
        results_df = pd.DataFrame(results)

        st.markdown("### Prediction Results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        n_delayed = sum(1 for r in results if r["Prediction"] == "DELAYED")
        total = len(results)

        if n_delayed >= 4:
            st.error(f"🚨 **{n_delayed} of {total}** algorithms predict **DELAYED**.")
        elif n_delayed >= 1:
            st.warning(f"⚠️ **{n_delayed} of {total}** algorithms predict **DELAYED**.")
        else:
            st.success(f"✅ All {total} algorithms predict **On time**.")

        st.caption(
            "⚠️ These are estimates based on historical patterns, not certainties. "
            "Actual delays depend on many factors not captured in the model."
        )
