#!/usr/bin/env python3
"""
SHAP Explainability Artifact Generator for LedgerGuard ML Models.

Generates SHAP-based explainability artifacts (summary beeswarm, bar plot, waterfall)
for all LedgerGuard ML models — three legacy Olist models and four industry models.

Legacy (Olist) models:
- Churn LightGBM       (models/churn/lightgbm_churn_model.pkl)
- Anomaly IsolationForest (models/anomaly/isolation_forest.joblib)
- Delivery XGBoost     (models/delivery/xgboost_late_delivery.joblib)

Industry models (primary focus):
- Churn Telco LightGBM (models/churn_industry/lightgbm_telco_churn.pkl)
- Delivery DataCo XGBoost (models/delivery_industry/xgboost_dataco_delivery.joblib)
- Sentiment LinearSVC  (models/sentiment_industry/linear_svc_financial_sentiment.joblib)
- Trend Forecaster LightGBM (models/trend/forecaster_refund_rate.joblib)

All plots are saved to reports/ and logged as MLflow artifacts under the
experiment "ledgerguard-model-explainability".

Usage:
    python scripts/generate_explainability.py --model all
    python scripts/generate_explainability.py --model churn_telco
    python scripts/generate_explainability.py --model delivery_dataco
    python scripts/generate_explainability.py --model sentiment
    python scripts/generate_explainability.py --model trend
    python scripts/generate_explainability.py --model churn
    python scripts/generate_explainability.py --model anomaly
    python scripts/generate_explainability.py --model delivery
"""

import argparse
import contextlib
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

# Keep project root on sys.path so internal imports (e.g. scripts.data_loader) resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data_loader import OlistDataLoader

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# XGBoost / SHAP version compatibility
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def shap_xgb_compat() -> Generator[None, None, None]:
    """
    Context manager that patches shap's internal UBJ decoder to handle the
    bracketed base_score format (e.g. ``[5E-1]``) emitted by some XGBoost 2.x
    model files when read by SHAP 0.49+.

    The incompatibility manifests as::

        ValueError: could not convert string to float: '[5E-1]'

    inside ``shap.explainers._tree.XGBTreeModelLoader.__init__``, which calls
    ``float(learner_model_param["base_score"])`` after decoding the UBJ binary
    payload.  The patch strips the surrounding brackets so the string is a
    valid Python float literal before the cast is attempted.

    The patch is applied for the duration of the ``with`` block only and
    restored unconditionally on exit, making it safe to use around a single
    ``shap.TreeExplainer(xgb_model)`` call.
    """
    try:
        original_decode = shap.explainers._tree.decode_ubjson_buffer

        def _safe_decode(fd: Any) -> Any:  # type: ignore[override]
            result = original_decode(fd)
            try:
                mp = result["learner"]["learner_model_param"]
                bs = mp.get("base_score", "")
                if isinstance(bs, str) and bs.startswith("["):
                    mp["base_score"] = re.sub(r"[\[\]]", "", bs).strip()
            except (KeyError, TypeError, AttributeError):
                pass
            return result

        shap.explainers._tree.decode_ubjson_buffer = _safe_decode
        yield
    finally:
        shap.explainers._tree.decode_ubjson_buffer = original_decode


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATHS = {
    # Legacy Olist models
    "churn": MODELS_DIR / "churn" / "lightgbm_churn_model.pkl",
    "anomaly": MODELS_DIR / "anomaly" / "isolation_forest.joblib",
    "delivery": MODELS_DIR / "delivery" / "xgboost_late_delivery.joblib",
    # Industry models
    "churn_telco": MODELS_DIR / "churn_industry" / "lightgbm_telco_churn.pkl",
    "delivery_dataco": MODELS_DIR / "delivery_industry" / "xgboost_dataco_delivery.joblib",
    "sentiment": MODELS_DIR / "sentiment_industry" / "linear_svc_financial_sentiment.joblib",
    "trend": MODELS_DIR / "trend" / "forecaster_refund_rate.joblib",
}

DATA_PATHS = {
    "churn_telco": DATA_DIR / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "delivery_dataco": DATA_DIR / "dataco" / "DataCoSupplyChainDataset.csv",
}

MLFLOW_EXPERIMENT = "ledgerguard-model-explainability"

# Maximum samples fed to SHAP to keep runtime manageable.
MAX_SHAP_SAMPLES = 500


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def log_ts(message: str) -> None:
    """Print a timestamped progress message to stdout."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def ensure_reports_dir() -> None:
    """Create the reports directory if it does not already exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def sample_data(
    X: pd.DataFrame, max_samples: int = MAX_SHAP_SAMPLES, random_state: int = 42
) -> pd.DataFrame:
    """
    Return up to *max_samples* rows from *X*, sampled without replacement.

    Args:
        X: Feature DataFrame to sample from.
        max_samples: Maximum number of rows to return.
        random_state: Seed for reproducibility.

    Returns:
        Sampled (or unchanged) DataFrame.
    """
    if len(X) <= max_samples:
        return X
    return X.sample(n=max_samples, random_state=random_state)


def sample_array(
    X: np.ndarray, max_samples: int = MAX_SHAP_SAMPLES, random_state: int = 42
) -> np.ndarray:
    """
    Return up to *max_samples* rows from a numpy array.

    Args:
        X: 2-D numpy array to sample from.
        max_samples: Maximum number of rows to return.
        random_state: Seed for reproducibility.

    Returns:
        Sampled (or unchanged) array.
    """
    if len(X) <= max_samples:
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx]


# ---------------------------------------------------------------------------
# Shared SHAP plot helpers
# ---------------------------------------------------------------------------


def save_summary_beeswarm(
    shap_values: Any,
    X_sample: Any,
    feature_names: List[str],
    name: str,
) -> Path:
    """
    Save a SHAP beeswarm summary plot.

    Args:
        shap_values: SHAP values array or Explanation object.
        X_sample: Feature matrix used to compute SHAP values.
        feature_names: Names of the features in *X_sample*.
        name: Short model identifier used in the filename.

    Returns:
        Absolute path of the saved PNG.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, _ = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Summary (beeswarm) — {name}", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = REPORTS_DIR / f"shap_summary_{name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_bar_plot(
    shap_values: Any,
    X_sample: Any,
    feature_names: List[str],
    name: str,
) -> Path:
    """
    Save a SHAP mean |SHAP| bar plot.

    Args:
        shap_values: SHAP values array or Explanation object.
        X_sample: Feature matrix used to compute SHAP values.
        feature_names: Names of the features in *X_sample*.
        name: Short model identifier used in the filename.

    Returns:
        Absolute path of the saved PNG.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, _ = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(
        f"SHAP Feature Importance (mean |SHAP|) — {name}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    out_path = REPORTS_DIR / f"shap_bar_{name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_waterfall(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_sample: Any,
    feature_names: List[str],
    name: str,
) -> Path:
    """
    Save a SHAP waterfall plot for the top-scoring prediction in the sample.

    The "top prediction" is the row with the highest absolute sum of SHAP
    values, which typically corresponds to the most confidently classified
    instance and produces the most informative waterfall chart.

    Args:
        explainer: Fitted shap.TreeExplainer.
        shap_values: 2-D SHAP values array (n_samples x n_features).
        X_sample: Feature DataFrame or ndarray aligned with *shap_values*.
        feature_names: Feature names for axis labels.
        name: Short model identifier used in the filename.

    Returns:
        Absolute path of the saved PNG.
    """
    abs_sums = np.abs(shap_values).sum(axis=1)
    top_idx = int(np.argmax(abs_sums))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, _ = plt.subplots(figsize=(12, 7))

    row_values = shap_values[top_idx]
    ev = explainer.expected_value
    if not isinstance(ev, (list, np.ndarray)):
        base_value = float(ev)
    elif np.asarray(ev).size == 1:
        base_value = float(np.asarray(ev).flat[0])
    else:
        base_value = float(ev[1])  # Binary classifier: positive class

    if isinstance(X_sample, pd.DataFrame):
        row_data = X_sample.iloc[top_idx].values
    else:
        row_data = X_sample[top_idx]

    exp_obj = shap.Explanation(
        values=row_values,
        base_values=base_value,
        data=row_data,
        feature_names=feature_names,
    )
    shap.plots.waterfall(exp_obj, show=False, max_display=15)
    plt.title(
        f"SHAP Waterfall (top prediction, idx={top_idx}) — {name}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    out_path = REPORTS_DIR / f"shap_waterfall_{name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Legacy Olist model explainability routines
# ---------------------------------------------------------------------------


def explain_churn(loader: OlistDataLoader) -> None:
    """
    Generate SHAP artifacts for the Olist LightGBM churn classifier.

    Loads model from models/churn/lightgbm_churn_model.pkl (a dict with
    keys 'model', 'threshold', and 'feature_names'), prepares churn data
    via OlistDataLoader, computes SHAP values with TreeExplainer, and
    saves three plots plus MLflow artifacts.

    Args:
        loader: Initialised OlistDataLoader.
    """
    name = "churn"
    model_path = MODEL_PATHS[name]

    log_ts(f"[{name}] Starting explainability generation")

    if not model_path.exists():
        print(f"WARNING: [{name}] Model file not found at {model_path} — skipping.")
        return

    log_ts(f"[{name}] Loading model from {model_path}")
    model_data = joblib.load(model_path)
    lgbm_model = model_data["model"]
    feature_names: List[str] = model_data.get("feature_names") or []

    log_ts(f"[{name}] Loading churn data via OlistDataLoader")
    X_train, _X_val, X_test, y_train, _y_val, y_test = loader.prepare_churn_data()

    if not feature_names:
        feature_names = list(X_test.columns)

    X_sample = sample_data(X_test, MAX_SHAP_SAMPLES)
    log_ts(f"[{name}] Using {len(X_sample)} samples for SHAP computation")

    log_ts(f"[{name}] Computing SHAP values with TreeExplainer")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    ensure_reports_dir()
    log_ts(f"[{name}] Saving beeswarm summary plot")
    summary_path = save_summary_beeswarm(shap_values, X_sample, feature_names, name)

    log_ts(f"[{name}] Saving bar plot")
    bar_path = save_bar_plot(shap_values, X_sample, feature_names, name)

    log_ts(f"[{name}] Saving waterfall plot")
    waterfall_path = save_waterfall(explainer, shap_values, X_sample, feature_names, name)

    log_ts(f"[{name}] Logging artifacts to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    with mlflow.start_run(run_name=f"explainability_{name}"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("model_path", str(model_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(waterfall_path))
        mlflow.log_param("shap_samples", len(X_sample))
        mlflow.log_param("n_features", len(feature_names))

    log_ts(f"[{name}] Done. Artifacts: {summary_path.name}, {bar_path.name}, {waterfall_path.name}")


def explain_anomaly(loader: OlistDataLoader) -> None:
    """
    Generate SHAP artifacts for the Olist Isolation Forest anomaly detector.

    Loads model from models/anomaly/isolation_forest.joblib (a raw
    IsolationForest instance), prepares anomaly data via OlistDataLoader,
    computes SHAP values with TreeExplainer, and saves three plots plus
    MLflow artifacts.

    Args:
        loader: Initialised OlistDataLoader.
    """
    name = "anomaly"
    model_path = MODEL_PATHS[name]

    log_ts(f"[{name}] Starting explainability generation")

    if not model_path.exists():
        print(f"WARNING: [{name}] Model file not found at {model_path} — skipping.")
        return

    log_ts(f"[{name}] Loading model from {model_path}")
    if_model = joblib.load(model_path)

    log_ts(f"[{name}] Loading anomaly data via OlistDataLoader")
    orders_df, order_items_df, reviews_df = loader.load_raw_data()
    df_daily = loader.aggregate_daily_metrics(orders_df, order_items_df, reviews_df)
    feature_cols: List[str] = list(df_daily.columns)

    X_all = df_daily[feature_cols]
    X_sample = sample_data(X_all, MAX_SHAP_SAMPLES)
    log_ts(f"[{name}] Using {len(X_sample)} samples for SHAP computation")

    log_ts(f"[{name}] Computing SHAP values with TreeExplainer (IsolationForest)")
    explainer = shap.TreeExplainer(if_model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    ensure_reports_dir()
    log_ts(f"[{name}] Saving beeswarm summary plot")
    summary_path = save_summary_beeswarm(shap_values, X_sample, feature_cols, name)

    log_ts(f"[{name}] Saving bar plot")
    bar_path = save_bar_plot(shap_values, X_sample, feature_cols, name)

    log_ts(f"[{name}] Saving waterfall plot")
    waterfall_path = save_waterfall(explainer, shap_values, X_sample, feature_cols, name)

    log_ts(f"[{name}] Logging artifacts to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    with mlflow.start_run(run_name=f"explainability_{name}"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("model_path", str(model_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(waterfall_path))
        mlflow.log_param("shap_samples", len(X_sample))
        mlflow.log_param("n_features", len(feature_cols))

    log_ts(f"[{name}] Done. Artifacts: {summary_path.name}, {bar_path.name}, {waterfall_path.name}")


def explain_delivery(loader: OlistDataLoader) -> None:
    """
    Generate SHAP artifacts for the Olist XGBoost late delivery classifier.

    Loads model from models/delivery/xgboost_late_delivery.joblib (a dict
    with keys 'model' and 'threshold'). The 'model' value may be either a
    raw XGBClassifier or a sklearn Pipeline with 'scaler' and 'classifier'
    named steps.

    For the Pipeline case the classifier is extracted and test features are
    scaled before being passed to SHAP. For the raw-model case features are
    passed directly.

    Args:
        loader: Initialised OlistDataLoader.
    """
    name = "delivery"
    model_path = MODEL_PATHS[name]

    log_ts(f"[{name}] Starting explainability generation")

    if not model_path.exists():
        print(f"WARNING: [{name}] Model file not found at {model_path} — skipping.")
        return

    log_ts(f"[{name}] Loading model from {model_path}")
    model_data = joblib.load(model_path)
    pipeline_or_model = model_data["model"]

    from sklearn.pipeline import Pipeline as SKPipeline

    if isinstance(pipeline_or_model, SKPipeline):
        log_ts(f"[{name}] Detected sklearn Pipeline — extracting classifier and scaler")
        xgb_classifier = pipeline_or_model.named_steps["classifier"]
        scaler = pipeline_or_model.named_steps["scaler"]
        uses_pipeline = True
    else:
        log_ts(f"[{name}] Model is a raw estimator (no pipeline wrapper)")
        xgb_classifier = pipeline_or_model
        scaler = None
        uses_pipeline = False

    log_ts(f"[{name}] Loading late delivery data via OlistDataLoader")
    X_train, _X_val, X_test, y_train, _y_val, y_test = loader.prepare_late_delivery_data()
    feature_names: List[str] = list(X_test.columns)

    X_sample = sample_data(X_test, MAX_SHAP_SAMPLES)
    log_ts(f"[{name}] Using {len(X_sample)} samples for SHAP computation")

    if uses_pipeline and scaler is not None:
        log_ts(f"[{name}] Applying pipeline scaler to sample features")
        X_shap: Any = scaler.transform(X_sample)
    else:
        X_shap = X_sample

    if isinstance(X_shap, np.ndarray):
        X_shap_display = pd.DataFrame(X_shap, columns=feature_names)
    else:
        X_shap_display = X_shap

    log_ts(f"[{name}] Computing SHAP values with TreeExplainer (XGBoost)")
    try:
        with shap_xgb_compat():
            explainer = shap.TreeExplainer(xgb_classifier)
        shap_values = explainer.shap_values(X_shap_display)
    except (ValueError, TypeError) as exc:
        log_ts(f"[{name}] TreeExplainer failed ({exc}) — falling back to KernelExplainer")
        X_bg = X_shap_display.iloc[:50]

        def _predict_proba_pos(x: np.ndarray) -> np.ndarray:
            return xgb_classifier.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(_predict_proba_pos, X_bg, nsamples=80)
        shap_values = explainer.shap_values(X_shap_display, nsamples=80)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    ensure_reports_dir()
    log_ts(f"[{name}] Saving beeswarm summary plot")
    summary_path = save_summary_beeswarm(shap_values, X_shap_display, feature_names, name)

    log_ts(f"[{name}] Saving bar plot")
    bar_path = save_bar_plot(shap_values, X_shap_display, feature_names, name)

    log_ts(f"[{name}] Saving waterfall plot")
    waterfall_path = save_waterfall(explainer, shap_values, X_shap_display, feature_names, name)

    log_ts(f"[{name}] Logging artifacts to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    with mlflow.start_run(run_name=f"explainability_{name}"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("model_path", str(model_path))
        mlflow.set_tag("uses_pipeline", str(uses_pipeline))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(waterfall_path))
        mlflow.log_param("shap_samples", len(X_sample))
        mlflow.log_param("n_features", len(feature_names))

    log_ts(f"[{name}] Done. Artifacts: {summary_path.name}, {bar_path.name}, {waterfall_path.name}")


# ---------------------------------------------------------------------------
# Industry model explainability routines (primary focus)
# ---------------------------------------------------------------------------


def _prepare_telco_features(csv_path: Path, feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply the same feature engineering used by TelcoChurnLoader to the raw CSV.

    This mirrors industry_data_loader.TelcoChurnLoader.load_and_prepare() so
    that SHAP is computed on identically processed features.

    Args:
        csv_path: Path to the Telco CSV file.
        feature_cols: Ordered list of feature column names (from model dict).

    Returns:
        DataFrame with one column per feature in *feature_cols*, NaN rows dropped.
    """
    df = pd.read_csv(csv_path)

    # Clean TotalCharges (blank strings for new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Target (needed for diagnostics only, not used in SHAP)
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)

    # Binary encode Yes/No columns
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col + "_encoded"] = (df[col] == "Yes").astype(int)

    df["SeniorCitizen_encoded"] = df["SeniorCitizen"]
    df["gender_encoded"] = (df["gender"] == "Male").astype(int)

    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[col + "_encoded"] = (df[col] == "Yes").astype(int)

    df["num_premium_services"] = sum(
        df[col + "_encoded"]
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    )

    df["MultipleLines_encoded"] = (df["MultipleLines"] == "Yes").astype(int)

    df["internet_dsl"] = (df["InternetService"] == "DSL").astype(int)
    df["internet_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["internet_none"] = (df["InternetService"] == "No").astype(int)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_encoded"] = df["Contract"].map(contract_map).fillna(0).astype(int)

    df["pay_electronic"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["pay_mailed"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    df["pay_bank_transfer"] = (df["PaymentMethod"] == "Bank transfer (automatic)").astype(int)
    df["pay_credit_card"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)

    df["tenure"] = df["tenure"].astype(float)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)

    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["charges_ratio"] = df["TotalCharges"] / (df["tenure"] * df["MonthlyCharges"] + 1)
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 100], labels=[0, 1, 2, 3]
    ).astype(int)
    df["high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
    df["at_risk_combo"] = (
        (df["contract_encoded"] == 0)
        & (df["internet_fiber"] == 1)
        & (df["TechSupport_encoded"] == 0)
    ).astype(int)

    # Keep only the feature columns present in the model artifact
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].dropna()
    return X


def _prepare_dataco_features(csv_path: Path, feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply the same feature engineering used by DataCoDeliveryLoader to the raw CSV.

    This mirrors industry_data_loader.DataCoDeliveryLoader.load_and_prepare() so
    that SHAP is computed on identically processed features.

    Args:
        csv_path: Path to the DataCo CSV file (latin-1 encoded).
        feature_cols: Ordered list of feature column names (from model dict).

    Returns:
        DataFrame with one column per feature in *feature_cols*, NaN rows dropped.
    """
    from sklearn.preprocessing import LabelEncoder

    try:
        df = pd.read_csv(csv_path, encoding="latin-1")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")

    # Target validation
    if "Late_delivery_risk" not in df.columns:
        df["Late_delivery_risk"] = (
            df["Days for shipping (real)"] > df["Days for shipment (scheduled)"]
        ).astype(int)

    # Temporal features
    if "order date (DateOrders)" in df.columns:
        df["order_date"] = pd.to_datetime(
            df["order date (DateOrders)"], format="mixed", dayfirst=False
        )
        df["order_day_of_week"] = df["order_date"].dt.dayofweek
        df["order_month"] = df["order_date"].dt.month
        df["order_hour"] = df["order_date"].dt.hour
        df["order_day_of_month"] = df["order_date"].dt.day
        df["is_weekend"] = (df["order_day_of_week"] >= 5).astype(int)
    else:
        for col, val in [
            ("order_day_of_week", 0), ("order_month", 1),
            ("order_hour", 12), ("order_day_of_month", 15), ("is_weekend", 0),
        ]:
            df[col] = val

    if "Days for shipment (scheduled)" in df.columns:
        df["scheduled_shipping_days"] = df["Days for shipment (scheduled)"]
    else:
        df["scheduled_shipping_days"] = 4

    order_value_map = {
        "Sales per customer": "sales_per_customer",
        "Order Item Discount": "item_discount",
        "Order Item Discount Rate": "discount_rate",
        "Order Item Profit Ratio": "profit_ratio",
        "Order Item Quantity": "item_quantity",
        "Sales": "sales",
        "Order Profit Per Order": "profit_per_order",
        "Order Item Product Price": "product_price",
    }
    for orig, new in order_value_map.items():
        if orig in df.columns:
            df[new] = pd.to_numeric(df[orig], errors="coerce").fillna(0)
        else:
            df[new] = 0

    if "Shipping Mode" in df.columns:
        ship_map = {"Standard Class": 0, "Second Class": 1, "First Class": 2, "Same Day": 3}
        df["shipping_mode_encoded"] = df["Shipping Mode"].map(ship_map).fillna(0).astype(int)
    else:
        df["shipping_mode_encoded"] = 0

    if "Customer Segment" in df.columns:
        seg_map = {"Consumer": 0, "Corporate": 1, "Home Office": 2}
        df["customer_segment_encoded"] = df["Customer Segment"].map(seg_map).fillna(0).astype(int)
    else:
        df["customer_segment_encoded"] = 0

    if "Market" in df.columns:
        le = LabelEncoder()
        df["market_encoded"] = le.fit_transform(df["Market"].fillna("Unknown"))
    else:
        df["market_encoded"] = 0

    if "Order Region" in df.columns:
        le = LabelEncoder()
        df["region_encoded"] = le.fit_transform(df["Order Region"].fillna("Unknown"))
    else:
        df["region_encoded"] = 0

    if "Category Name" in df.columns:
        le = LabelEncoder()
        df["category_encoded"] = le.fit_transform(df["Category Name"].fillna("Unknown"))
    elif "Department Name" in df.columns:
        le = LabelEncoder()
        df["category_encoded"] = le.fit_transform(df["Department Name"].fillna("Unknown"))
    else:
        df["category_encoded"] = 0

    if "Department Name" in df.columns:
        le = LabelEncoder()
        df["department_encoded"] = le.fit_transform(df["Department Name"].fillna("Unknown"))
    else:
        df["department_encoded"] = 0

    if "Order Status" in df.columns:
        pre_shipment_map = {
            "PENDING": 1, "PROCESSING": 2, "PENDING_PAYMENT": 3,
            "ON_HOLD": 4, "SUSPECTED_FRAUD": 5, "PAYMENT_REVIEW": 6,
        }
        df["order_status_encoded"] = df["Order Status"].map(pre_shipment_map).fillna(0).astype(int)
    else:
        df["order_status_encoded"] = 0

    df["value_per_item"] = df["sales"] / (df["item_quantity"] + 1)
    df["discount_amount"] = df["sales"] * df["discount_rate"]
    df["high_value_order"] = (df["sales"] > df["sales"].quantile(0.75)).astype(int)
    df["rush_shipping"] = (df["shipping_mode_encoded"] >= 2).astype(int)
    df["high_quantity"] = (df["item_quantity"] > df["item_quantity"].quantile(0.75)).astype(int)
    df["shipping_schedule_interaction"] = (
        df["shipping_mode_encoded"] * df["scheduled_shipping_days"]
    )
    df["tight_schedule"] = (
        (df["shipping_mode_encoded"] == 0) & (df["scheduled_shipping_days"] <= 3)
    ).astype(int)
    df["high_risk_combo"] = (
        (df["shipping_mode_encoded"] == 0)
        & (df["customer_segment_encoded"] == 0)
        & (df["scheduled_shipping_days"] >= 4)
    ).astype(int)

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)

    # Drop rows where target is NaN (mirrors loader behaviour)
    mask = df["Late_delivery_risk"].notna()
    X = X[mask]
    return X


def explain_churn_telco(_loader: Optional[OlistDataLoader] = None) -> None:
    """
    Generate SHAP artifacts for the Telco LightGBM churn classifier.

    Model:   models/churn_industry/lightgbm_telco_churn.pkl
             Dict keys: 'model' (LGBMClassifier), 'threshold', 'features'
    Data:    data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv
    Outputs: shap_summary_churn_telco.png, shap_bar_churn_telco.png,
             shap_waterfall_churn_telco.png

    Args:
        _loader: Unused; present for uniform dispatch signature.
    """
    model_key = "churn_telco"
    artifact_name = "churn_telco"
    model_path = MODEL_PATHS[model_key]
    data_path = DATA_PATHS[model_key]

    log_ts(f"[{model_key}] Starting explainability generation")

    if not model_path.exists():
        print(f"WARNING: [{model_key}] Model not found at {model_path} — skipping.")
        return

    if not data_path.exists():
        print(
            f"WARNING: [{model_key}] Data file not found at {data_path} — skipping. "
            "Run: python scripts/download_datasets.py --dataset telco"
        )
        return

    log_ts(f"[{model_key}] Loading model from {model_path}")
    model_data = joblib.load(model_path)
    lgbm_model = model_data["model"]
    feature_cols: List[str] = list(model_data["features"])

    log_ts(f"[{model_key}] Loading and engineering Telco features from {data_path}")
    X = _prepare_telco_features(data_path, feature_cols)
    log_ts(f"[{model_key}] Loaded {len(X)} rows with {len(feature_cols)} features")

    X_sample = sample_data(X, MAX_SHAP_SAMPLES)
    log_ts(f"[{model_key}] Using {len(X_sample)} samples for SHAP computation")

    log_ts(f"[{model_key}] Computing SHAP values with TreeExplainer (LightGBM)")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary: may return list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    ensure_reports_dir()
    log_ts(f"[{model_key}] Saving beeswarm summary plot")
    summary_path = save_summary_beeswarm(shap_values, X_sample, feature_cols, artifact_name)

    log_ts(f"[{model_key}] Saving bar plot")
    bar_path = save_bar_plot(shap_values, X_sample, feature_cols, artifact_name)

    log_ts(f"[{model_key}] Saving waterfall plot")
    waterfall_path = save_waterfall(explainer, shap_values, X_sample, feature_cols, artifact_name)

    log_ts(f"[{model_key}] Logging artifacts to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    with mlflow.start_run(run_name=f"explainability_{model_key}"):
        mlflow.set_tag("model_name", model_key)
        mlflow.set_tag("model_path", str(model_path))
        mlflow.set_tag("dataset", "IBM_Telco")
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(waterfall_path))
        mlflow.log_param("shap_samples", len(X_sample))
        mlflow.log_param("n_features", len(feature_cols))

    log_ts(
        f"[{model_key}] Done. Artifacts: {summary_path.name}, {bar_path.name}, {waterfall_path.name}"
    )


def explain_delivery_dataco(_loader: Optional[OlistDataLoader] = None) -> None:
    """
    Generate SHAP artifacts for the DataCo XGBoost late-delivery classifier.

    Model:   models/delivery_industry/xgboost_dataco_delivery.joblib
             Dict keys: 'model' (XGBClassifier — raw, not a Pipeline),
             'threshold', 'features'
    Data:    data/dataco/DataCoSupplyChainDataset.csv
    Outputs: shap_summary_delivery_dataco.png, shap_bar_delivery_dataco.png,
             shap_waterfall_delivery_dataco.png

    Note: The saved artifact stores a raw XGBClassifier (not wrapped in a
    sklearn Pipeline), so no scaler step is needed. SHAP TreeExplainer works
    directly on the classifier.

    Args:
        _loader: Unused; present for uniform dispatch signature.
    """
    model_key = "delivery_dataco"
    artifact_name = "delivery_dataco"
    model_path = MODEL_PATHS[model_key]
    data_path = DATA_PATHS[model_key]

    log_ts(f"[{model_key}] Starting explainability generation")

    if not model_path.exists():
        print(f"WARNING: [{model_key}] Model not found at {model_path} — skipping.")
        return

    if not data_path.exists():
        print(
            f"WARNING: [{model_key}] Data file not found at {data_path} — skipping. "
            "Run: python scripts/download_datasets.py --dataset dataco"
        )
        return

    log_ts(f"[{model_key}] Loading model from {model_path}")
    model_data = joblib.load(model_path)
    feature_cols: List[str] = list(model_data["features"])

    # The stored artifact holds a raw XGBClassifier (confirmed at generation time).
    # Guard against future Pipeline wrapping with an explicit branch.
    from sklearn.pipeline import Pipeline as SKPipeline

    raw_model = model_data["model"]
    if isinstance(raw_model, SKPipeline):
        log_ts(f"[{model_key}] Detected sklearn Pipeline — extracting classifier step")
        xgb_clf = raw_model.named_steps["classifier"]
        scaler_step = raw_model.named_steps.get("scaler")
    else:
        xgb_clf = raw_model
        scaler_step = None

    log_ts(f"[{model_key}] Loading and engineering DataCo features from {data_path}")
    X = _prepare_dataco_features(data_path, feature_cols)
    log_ts(f"[{model_key}] Loaded {len(X)} rows with {len(feature_cols)} features")

    X_sample = sample_data(X, MAX_SHAP_SAMPLES)
    log_ts(f"[{model_key}] Using {len(X_sample)} samples for SHAP computation")

    # Apply scaler only if the model was wrapped in a Pipeline with a scaler.
    if scaler_step is not None:
        log_ts(f"[{model_key}] Applying pipeline scaler")
        X_shap_arr = scaler_step.transform(X_sample[feature_cols])
        X_shap: Any = pd.DataFrame(X_shap_arr, columns=feature_cols)
    else:
        X_shap = X_sample[feature_cols]

    log_ts(f"[{model_key}] Computing SHAP values with TreeExplainer (XGBoost)")
    # Use shap_xgb_compat() to handle the bracketed base_score format ([5E-1])
    # that some XGBoost 2.x artifacts produce when parsed by SHAP 0.49+.
    with shap_xgb_compat():
        explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    ensure_reports_dir()
    log_ts(f"[{model_key}] Saving beeswarm summary plot")
    summary_path = save_summary_beeswarm(shap_values, X_shap, feature_cols, artifact_name)

    log_ts(f"[{model_key}] Saving bar plot")
    bar_path = save_bar_plot(shap_values, X_shap, feature_cols, artifact_name)

    log_ts(f"[{model_key}] Saving waterfall plot")
    waterfall_path = save_waterfall(explainer, shap_values, X_shap, feature_cols, artifact_name)

    log_ts(f"[{model_key}] Logging artifacts to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    with mlflow.start_run(run_name=f"explainability_{model_key}"):
        mlflow.set_tag("model_name", model_key)
        mlflow.set_tag("model_path", str(model_path))
        mlflow.set_tag("dataset", "DataCo_Supply_Chain")
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(waterfall_path))
        mlflow.log_param("shap_samples", len(X_sample))
        mlflow.log_param("n_features", len(feature_cols))

    log_ts(
        f"[{model_key}] Done. Artifacts: {summary_path.name}, {bar_path.name}, {waterfall_path.name}"
    )


def explain_sentiment(_loader: Optional[OlistDataLoader] = None) -> None:
    """
    Generate coefficient-importance artifacts for the financial sentiment LinearSVC.

    SHAP is impractical for TF-IDF + LinearSVC (sparse, very high-dimensional).
    Instead this function extracts the raw LinearSVC coefficients from inside
    the CalibratedClassifierCV wrapper and plots:
    - shap_summary_sentiment.png  — top-25 words by mean |coef| across all classes
    - shap_bar_sentiment.png      — identical plot saved under the bar filename
                                    (UI expects both names for all models)
    - shap_waterfall_sentiment.png — per-class top-10 coefficients as a
                                     grouped horizontal bar chart

    Model path: models/sentiment_industry/linear_svc_financial_sentiment.joblib
    Model type: sklearn Pipeline with steps 'tfidf' (TfidfVectorizer) and
                'clf' (CalibratedClassifierCV wrapping LinearSVC).

    The CalibratedClassifierCV exposes fitted LinearSVC instances via:
        pipeline.named_steps['clf'].calibrated_classifiers_[i].estimator.coef_
    Each inner LinearSVC has coef_ shape (n_classes, n_features) = (3, 2411).
    We average across the cross-validation folds and then across classes to
    obtain a single importance score per vocabulary token.

    Classes: 0 = negative, 1 = neutral, 2 = positive.

    Args:
        _loader: Unused; present for uniform dispatch signature.
    """
    model_key = "sentiment"
    artifact_name = "sentiment"
    model_path = MODEL_PATHS[model_key]

    log_ts(f"[{model_key}] Starting coefficient-importance generation (non-SHAP)")

    if not model_path.exists():
        print(f"WARNING: [{model_key}] Model not found at {model_path} — skipping.")
        return

    log_ts(f"[{model_key}] Loading pipeline from {model_path}")
    pipeline = joblib.load(model_path)

    # --- Extract TF-IDF vocabulary ---
    tfidf = pipeline.named_steps["tfidf"]
    feature_names_arr = tfidf.get_feature_names_out()
    feature_names = list(feature_names_arr)
    n_features = len(feature_names)
    log_ts(f"[{model_key}] TF-IDF vocabulary size: {n_features}")

    # --- Extract LinearSVC coefficients from CalibratedClassifierCV ---
    calibrated_cv = pipeline.named_steps["clf"]
    # calibrated_classifiers_ is a list of _CalibratedClassifier objects,
    # one per CV fold. Each exposes .estimator (the fitted LinearSVC).
    coef_folds: List[np.ndarray] = []
    for cal_clf in calibrated_cv.calibrated_classifiers_:
        inner_svc = cal_clf.estimator
        if hasattr(inner_svc, "coef_"):
            coef_folds.append(inner_svc.coef_)  # shape: (n_classes, n_features)

    if not coef_folds:
        print(
            f"WARNING: [{model_key}] Could not extract coef_ from LinearSVC internals — skipping."
        )
        return

    # Average across CV folds -> (n_classes, n_features)
    coef_mean = np.mean(coef_folds, axis=0)
    n_classes = coef_mean.shape[0]
    class_names = ["negative", "neutral", "positive"][:n_classes]

    log_ts(
        f"[{model_key}] Extracted coefficients: shape {coef_mean.shape} "
        f"({len(coef_folds)} CV folds averaged)"
    )

    # --- Plot 1: top-25 words by mean |coef| across all classes ---
    # (saved as both shap_summary_sentiment.png and shap_bar_sentiment.png)
    mean_abs_coef = np.abs(coef_mean).mean(axis=0)  # (n_features,)
    top25_idx = np.argsort(mean_abs_coef)[-25:][::-1]
    top25_words = [feature_names[i] for i in top25_idx]
    top25_importances = mean_abs_coef[top25_idx]

    def _plot_top25_bar(out_name: str) -> Path:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.RdYlGn(
            np.linspace(0.85, 0.15, len(top25_words))
        )
        bars = ax.barh(
            range(len(top25_words)),
            top25_importances[::-1],
            color=colors[::-1],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_yticks(range(len(top25_words)))
        ax.set_yticklabels(top25_words[::-1], fontsize=10)
        ax.set_xlabel("Mean |Coefficient| across classes", fontsize=11)
        ax.set_title(
            "Top 25 Most Influential Words (LinearSVC Coefficients)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.invert_yaxis()
        plt.tight_layout()
        path = REPORTS_DIR / f"{out_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    ensure_reports_dir()
    log_ts(f"[{model_key}] Saving summary importance plot")
    summary_path = _plot_top25_bar(f"shap_summary_{artifact_name}")

    log_ts(f"[{model_key}] Saving bar importance plot (same as summary for UI consistency)")
    bar_path = _plot_top25_bar(f"shap_bar_{artifact_name}")

    # --- Plot 2: per-class top-10 coefficients (waterfall equivalent) ---
    log_ts(f"[{model_key}] Saving per-class waterfall plot")

    n_top = 10
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 7), sharey=False)
    if n_classes == 1:
        axes = [axes]

    palette = ["#e74c3c", "#95a5a6", "#27ae60"]

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, class_names)):
        cls_coef = coef_mean[cls_idx]
        top_pos_idx = np.argsort(cls_coef)[-n_top:][::-1]
        top_neg_idx = np.argsort(cls_coef)[:n_top]

        # Take the union: top positive + top negative drivers
        combined_idx = list(top_pos_idx) + list(top_neg_idx)
        # Deduplicate preserving order
        seen: set = set()
        unique_idx: List[int] = []
        for i in combined_idx:
            if i not in seen:
                seen.add(i)
                unique_idx.append(i)
        unique_idx = unique_idx[:n_top]

        words_cls = [feature_names[i] for i in unique_idx]
        coefs_cls = cls_coef[unique_idx]

        bar_colors = [palette[cls_idx] if v >= 0 else "#bdc3c7" for v in coefs_cls]
        ax.barh(
            range(len(words_cls)),
            coefs_cls,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_yticks(range(len(words_cls)))
        ax.set_yticklabels(words_cls, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{cls_name.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=10)
        ax.invert_yaxis()

    plt.suptitle(
        "LinearSVC Per-Class Top Coefficients — Financial Sentiment",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    waterfall_path = REPORTS_DIR / f"shap_waterfall_{artifact_name}.png"
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log_ts(f"[{model_key}] Logging artifacts to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    with mlflow.start_run(run_name=f"explainability_{model_key}"):
        mlflow.set_tag("model_name", model_key)
        mlflow.set_tag("model_path", str(model_path))
        mlflow.set_tag("method", "LinearSVC_coefficients")
        mlflow.set_tag("dataset", "FinancialPhraseBank")
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(waterfall_path))
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("n_cv_folds", len(coef_folds))
        mlflow.log_param("top_k_words", 25)

    log_ts(
        f"[{model_key}] Done. Artifacts: {summary_path.name}, {bar_path.name}, {waterfall_path.name}"
    )


def explain_trend(_loader: Optional[OlistDataLoader] = None) -> None:
    """
    Generate SHAP artifacts for the LightGBM trend forecaster (refund_rate).

    The trend forecaster operates on lag-based features (no real feature
    matrix from a dataset exists at inference time). Synthetic X is generated
    as random values in [0.0, 0.2] — representative of the refund_rate domain —
    so that SHAP TreeExplainer can produce meaningful lag-importance rankings.

    Model:   models/trend/forecaster_refund_rate.joblib
             Dict keys: 'model' (LGBMRegressor), 'meta' (dict with n_lags,
             n_features, metric_name, etc.)
    Features (19 total):
             lag_0 .. lag_13  (14 lag values)
             rolling_7_mean, rolling_7_std, rolling_14_mean, rolling_14_std,
             trend_delta
    Outputs: shap_summary_trend.png, shap_bar_trend.png,
             shap_waterfall_trend.png

    Args:
        _loader: Unused; present for uniform dispatch signature.
    """
    model_key = "trend"
    artifact_name = "trend"
    model_path = MODEL_PATHS[model_key]

    log_ts(f"[{model_key}] Starting explainability generation")

    if not model_path.exists():
        print(f"WARNING: [{model_key}] Model not found at {model_path} — skipping.")
        return

    log_ts(f"[{model_key}] Loading model from {model_path}")
    model_data = joblib.load(model_path)
    lgbm_regressor = model_data["model"]
    meta: dict = model_data.get("meta", {})

    n_lags: int = meta.get("n_lags", 14)
    n_features: int = meta.get("n_features", 19)
    metric_name: str = meta.get("metric_name", "refund_rate")

    # Build ordered feature names to match training layout:
    # [lag_0, lag_1, ..., lag_{n_lags-1}, rolling_7_mean, rolling_7_std,
    #  rolling_14_mean, rolling_14_std, trend_delta]
    feature_names: List[str] = [f"lag_{i}" for i in range(n_lags)] + [
        "rolling_7_mean",
        "rolling_7_std",
        "rolling_14_mean",
        "rolling_14_std",
        "trend_delta",
    ]

    # Validate expected count
    if len(feature_names) != n_features:
        # Fallback: use generic names if meta indicates a different count
        feature_names = [f"feature_{i}" for i in range(n_features)]
        log_ts(
            f"[{model_key}] WARNING: expected {n_features} features but computed "
            f"{len(feature_names)} named features — using generic names"
        )

    log_ts(
        f"[{model_key}] Generating synthetic X: 200 samples x {n_features} features "
        f"(range [0.0, 0.2] — representative {metric_name} domain)"
    )
    rng = np.random.default_rng(42)
    X_synthetic = rng.random((200, n_features)) * 0.2  # shape (200, n_features)
    X_sample_df = pd.DataFrame(X_synthetic, columns=feature_names)

    log_ts(f"[{model_key}] Computing SHAP values with TreeExplainer (LGBMRegressor)")
    explainer = shap.TreeExplainer(lgbm_regressor)
    shap_values = explainer.shap_values(X_sample_df)

    # LGBMRegressor returns a single 2-D array (no list wrapping)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    ensure_reports_dir()
    log_ts(f"[{model_key}] Saving beeswarm summary plot")
    summary_path = save_summary_beeswarm(
        shap_values, X_sample_df, feature_names, artifact_name
    )

    log_ts(f"[{model_key}] Saving bar plot")
    bar_path = save_bar_plot(shap_values, X_sample_df, feature_names, artifact_name)

    log_ts(f"[{model_key}] Saving waterfall plot")
    waterfall_path = save_waterfall(
        explainer, shap_values, X_sample_df, feature_names, artifact_name
    )

    log_ts(f"[{model_key}] Logging artifacts to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    with mlflow.start_run(run_name=f"explainability_{model_key}"):
        mlflow.set_tag("model_name", model_key)
        mlflow.set_tag("model_path", str(model_path))
        mlflow.set_tag("metric_name", metric_name)
        mlflow.set_tag("data_source", "synthetic_representative")
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(waterfall_path))
        mlflow.log_param("shap_samples", len(X_sample_df))
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("n_lags", n_lags)
        mlflow.log_param("synthetic_range", "[0.0, 0.2]")
        if meta:
            mlflow.log_param("train_mae", meta.get("train_mae", "n/a"))
            mlflow.log_param("test_mae", meta.get("test_mae", "n/a"))

    log_ts(
        f"[{model_key}] Done. Artifacts: {summary_path.name}, {bar_path.name}, {waterfall_path.name}"
    )


# ---------------------------------------------------------------------------
# Model dispatch table
# ---------------------------------------------------------------------------

# Industry models listed first so they are the primary focus when running --model all.
# The OlistDataLoader argument is passed for all handlers; industry handlers ignore it.
EXPLAIN_HANDLERS = {
    # Industry models (primary)
    "churn_telco": explain_churn_telco,
    "delivery_dataco": explain_delivery_dataco,
    "sentiment": explain_sentiment,
    "trend": explain_trend,
    # Legacy Olist models (secondary, kept for backward compatibility)
    "churn": explain_churn,
    "anomaly": explain_anomaly,
    "delivery": explain_delivery,
}

# Canonical execution order when --model all is requested
ALL_MODEL_KEYS: List[str] = [
    "churn_telco",
    "delivery_dataco",
    "sentiment",
    "trend",
    "churn",
    "anomaly",
    "delivery",
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explainability artifacts for LedgerGuard ML models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Industry models (primary focus):
  python scripts/generate_explainability.py --model churn_telco
  python scripts/generate_explainability.py --model delivery_dataco
  python scripts/generate_explainability.py --model sentiment
  python scripts/generate_explainability.py --model trend

Legacy Olist models:
  python scripts/generate_explainability.py --model churn
  python scripts/generate_explainability.py --model anomaly
  python scripts/generate_explainability.py --model delivery

All models (industry first, then legacy):
  python scripts/generate_explainability.py --model all
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(EXPLAIN_HANDLERS.keys()) + ["all"],
        default="all",
        help="Which model to explain. Use 'all' to run all 7 models (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    models_to_run: List[str] = ALL_MODEL_KEYS if args.model == "all" else [args.model]

    log_ts("=" * 72)
    log_ts("LEDGERGUARD SHAP EXPLAINABILITY GENERATOR")
    log_ts("=" * 72)
    log_ts(f"Models to explain : {', '.join(models_to_run)}")
    log_ts(f"Output directory  : {REPORTS_DIR}")
    log_ts(f"MLflow experiment : {MLFLOW_EXPERIMENT}")
    log_ts("=" * 72)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Initialise OlistDataLoader once. Legacy handlers call the specific method
    # they need; industry handlers receive it but ignore it.
    loader = OlistDataLoader()

    failed: List[str] = []

    for model_name in models_to_run:
        log_ts(f"\n--- Processing model: {model_name} ---")
        try:
            EXPLAIN_HANDLERS[model_name](loader)
        except Exception as exc:
            print(f"WARNING: [{model_name}] Failed with error: {exc}")
            import traceback
            traceback.print_exc()
            failed.append(model_name)

    log_ts("\n" + "=" * 72)
    log_ts("EXPLAINABILITY GENERATION COMPLETE")
    log_ts("=" * 72)

    succeeded = [m for m in models_to_run if m not in failed]
    if succeeded:
        log_ts(f"Succeeded : {', '.join(succeeded)}")
    if failed:
        log_ts(f"Failed    : {', '.join(failed)}")

    log_ts(f"\nArtifacts saved to : {REPORTS_DIR.resolve()}")
    log_ts("View in MLflow     : mlflow ui  (http://localhost:5000)")


if __name__ == "__main__":
    main()
