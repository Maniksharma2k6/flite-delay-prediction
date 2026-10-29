"""
Shared utilities for the Flight Delay Prediction app.
Column lists, preprocessing, model builder, training, metrics, save/load.
Now includes Feature Selection, AdaBoost, and Ensemble (Voting).
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "hour", "distance", "temp", "dewp", "humid",
    "wind_dir", "wind_speed", "precip", "pressure", "visib",
]

CATEGORICAL_COLS = ["month", "dow", "carrier", "origin", "dest"]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

TARGET_COL = "delayed"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
BUNDLE_PATH = os.path.join(MODEL_DIR, "bundle.joblib")
HOLDOUT_PATH = os.path.join(MODEL_DIR, "holdout_test.csv")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "flights_sample.csv")

ALGORITHM_NAMES = [
    "Logistic Regression",
    "K-Nearest Neighbours",
    "Support Vector Machine",
    "Random Forest",
    "AdaBoost",
    "Ensemble (Voting)",
]

FS_METHODS = {
    "ANOVA F-test (f_classif)": f_classif,
    "Mutual Information": mutual_info_classif,
}

# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV, cast month & dow to str for one-hot encoding."""
    df = pd.read_csv(path)
    for col in ["month", "dow"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def validate_columns(df: pd.DataFrame, need_target: bool = True):
    """Return list of missing required columns (empty list == OK)."""
    required = set(FEATURE_COLS)
    if need_target:
        required.add(TARGET_COL)
    return sorted(required - set(df.columns))


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ],
        sparse_threshold=0,
    )


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_models(k_best: int | None = None, fs_method_name: str = "ANOVA F-test (f_classif)"):
    """
    Return an ordered dict of {name: Pipeline}.
    Each pipeline gets its own fresh preprocessor (and optional feature selector).
    """
    score_func = FS_METHODS.get(fs_method_name, f_classif)

    def _make_steps():
        steps = [("prep", build_preprocessor())]
        if k_best is not None and k_best > 0:
            steps.append(("select", SelectKBest(score_func, k=k_best)))
        return steps

    models = {}

    # 1. Logistic Regression
    models["Logistic Regression"] = Pipeline(_make_steps() + [
        ("clf", LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    # 2. KNN
    models["K-Nearest Neighbours"] = Pipeline(_make_steps() + [
        ("clf", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        )),
    ])

    # 3. SVM (no probability – uses decision_function for ROC-AUC)
    models["Support Vector Machine"] = Pipeline(_make_steps() + [
        ("clf", SVC(
            kernel="rbf",
            C=1,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # 4. Random Forest
    models["Random Forest"] = Pipeline(_make_steps() + [
        ("clf", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    # 5. AdaBoost
    models["AdaBoost"] = Pipeline(_make_steps() + [
        ("clf", AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        )),
    ])

    # 6. Ensemble – soft voting with the 4 models that support predict_proba
    #    (SVM excluded because probability=True is too slow)
    models["Ensemble (Voting)"] = Pipeline(_make_steps() + [
        ("clf", VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(
                    C=0.1, class_weight="balanced",
                    max_iter=1000, random_state=42)),
                ("knn", KNeighborsClassifier(
                    n_neighbors=5, weights="distance")),
                ("rf", RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=5,
                    class_weight="balanced", n_jobs=-1, random_state=42)),
                ("ada", AdaBoostClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42)),
            ],
            voting="soft",
        )),
    ])

    return models


# ---------------------------------------------------------------------------
# Feature importance helpers
# ---------------------------------------------------------------------------

def get_feature_importance(bundle: dict):
    """
    Extract feature importance from the Random Forest model.
    Returns (aggregated_df, raw_df):
        aggregated_df – importance per *original* feature (categoricals summed)
        raw_df        – importance per *transformed* feature (top 25)
    """
    rf_pipe = bundle["fitted"]["Random Forest"]
    preprocessor = rf_pipe.named_steps["prep"]
    clf = rf_pipe.named_steps["clf"]

    feature_names = preprocessor.get_feature_names_out()

    # If feature selection was used, filter to selected features
    if "select" in rf_pipe.named_steps:
        mask = rf_pipe.named_steps["select"].get_support()
        feature_names = feature_names[mask]

    importances = clf.feature_importances_

    # --- Raw (transformed feature level) ---
    raw_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )

    # --- Aggregated to original feature level ---
    agg = {}
    for fname, imp in zip(feature_names, importances):
        if "__" in str(fname):
            prefix, rest = str(fname).split("__", 1)
            if prefix == "num":
                orig = rest
            elif prefix == "cat":
                orig = rest  # fallback
                for cat_col in CATEGORICAL_COLS:
                    if rest == cat_col or rest.startswith(cat_col + "_"):
                        orig = cat_col
                        break
            else:
                orig = str(fname)
        else:
            orig = str(fname)
        agg[orig] = agg.get(orig, 0) + imp

    agg_df = (
        pd.DataFrame([{"Feature": k, "Importance": v} for k, v in agg.items()])
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return agg_df, raw_df


def get_selectkbest_scores(bundle: dict):
    """
    If feature selection was used, return a DataFrame of all features and
    their SelectKBest scores.  Returns None if feature selection was not used.
    """
    rf_pipe = bundle["fitted"]["Random Forest"]
    if "select" not in rf_pipe.named_steps:
        return None

    preprocessor = rf_pipe.named_steps["prep"]
    selector = rf_pipe.named_steps["select"]
    feature_names = preprocessor.get_feature_names_out()
    scores = selector.scores_

    df = (
        pd.DataFrame({"Feature": feature_names, "Score": scores})
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )
    df["Selected"] = selector.get_support()
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all(
    df: pd.DataFrame,
    test_size: float = 0.2,
    k_best: int | None = None,
    fs_method_name: str = "ANOVA F-test (f_classif)",
    progress_cb=None,
):
    """
    Train all six models and return results dict.
    progress_cb(fraction, message) is called to update a Streamlit progress bar.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42,
    )

    models = build_models(k_best=k_best, fs_method_name=fs_method_name)

    fitted = {}
    metrics_rows = []
    cm_dict = {}
    roc_dict = {}

    n_models = len(models)
    for i, (name, pipe) in enumerate(models.items()):
        if progress_cb:
            progress_cb(i / n_models, f"Training {name}\u2026")

        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        y_pred = pipe.predict(X_test)

        # ROC-AUC: use decision_function for SVM, predict_proba for others
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_score = pipe.predict_proba(X_test)[:, 1]
        else:
            y_score = pipe.decision_function(X_test)

        auc = roc_auc_score(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_dict[name] = (fpr, tpr, auc)

        metrics_rows.append({
            "Algorithm": name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC":   round(auc, 4),
        })

        cm_dict[name] = confusion_matrix(y_test, y_pred)

    if progress_cb:
        progress_cb(1.0, "Done!")

    metrics_df = pd.DataFrame(metrics_rows)

    # Compute median defaults from training set (for single-flight form)
    median_defaults = {}
    for col in NUMERIC_COLS:
        median_defaults[col] = float(X_train[col].median())

    # Category lists (for dropdowns)
    cat_lists = {}
    for col in CATEGORICAL_COLS:
        cat_lists[col] = sorted(X_train[col].dropna().unique().tolist())

    # Build test DataFrame to save
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    return {
        "fitted": fitted,
        "metrics_df": metrics_df,
        "cm_dict": cm_dict,
        "roc_dict": roc_dict,
        "median_defaults": median_defaults,
        "cat_lists": cat_lists,
        "test_df": test_df,
        "feature_selection": {
            "enabled": k_best is not None and k_best > 0,
            "k_best": k_best,
            "method": fs_method_name,
        },
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_bundle(results: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "fitted": results["fitted"],
        "metrics_df": results["metrics_df"],
        "cm_dict": results["cm_dict"],
        "roc_dict": results["roc_dict"],
        "median_defaults": results["median_defaults"],
        "cat_lists": results["cat_lists"],
        "feature_selection": results.get("feature_selection", {}),
    }, BUNDLE_PATH)
    results["test_df"].to_csv(HOLDOUT_PATH, index=False)


def load_bundle() -> dict | None:
    if not os.path.exists(BUNDLE_PATH):
        return None
    return joblib.load(BUNDLE_PATH)


def bundle_exists() -> bool:
    return os.path.exists(BUNDLE_PATH)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_single(bundle: dict, input_dict: dict):
    """
    Predict for a single flight across all six algorithms.
    input_dict keys must match FEATURE_COLS.
    Returns list of dicts with keys: Algorithm, Prediction, Probability.
    """
    row = pd.DataFrame([input_dict])
    for col in ["month", "dow"]:
        row[col] = row[col].astype(str)

    results = []
    for name, pipe in bundle["fitted"].items():
        pred = pipe.predict(row)[0]
        label = "DELAYED" if pred == 1 else "On time"

        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            prob = pipe.predict_proba(row)[0][1]
            prob_str = f"{prob:.1%}"
        else:
            prob_str = "N/A (SVM)"

        results.append({
            "Algorithm": name,
            "Prediction": label,
            "Delay Probability": prob_str,
        })
    return results


def predict_batch(pipe, X: pd.DataFrame):
    """
    Return predictions and scores for a batch.
    Scores are probabilities where available, else decision_function values.
    """
    preds = pipe.predict(X)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        scores = pipe.predict_proba(X)[:, 1]
    else:
        scores = pipe.decision_function(X)
    return preds, scores


# ---------------------------------------------------------------------------
# Metric computation on new data
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_score), 4),
    }
