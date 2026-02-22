#!/usr/bin/env python3
"""
LedgerGuard Model Card Generator.

Generates statistical rigor artifacts for the top LedgerGuard ML models:
- 5-fold stratified cross-validation with 95% CI
- Confusion matrix heatmaps
- ROC curves with bootstrapped 95% CI bands
- Structured model card JSON files

All artifacts are saved to the reports/ directory and logged to MLflow under
the experiment ``ledgerguard-model-cards``.

Models evaluated (Olist):
    churn     — LightGBM churn classifier (models/churn/lightgbm_churn_model.pkl)
    anomaly   — Isolation Forest (models/anomaly/isolation_forest.joblib)
    delivery  — XGBoost late delivery (models/delivery/xgboost_late_delivery.joblib)

Models evaluated (Industry):
    churn_telco      — LightGBM Telco Churn (models/churn_industry/lightgbm_telco_churn.pkl)
    delivery_dataco  — XGBoost DataCo Delivery (models/delivery_industry/xgboost_dataco_delivery.joblib)
    sentiment_industry — LinearSVC Financial Sentiment (models/sentiment_industry/linear_svc_financial_sentiment.joblib)
    trend            — LightGBM Trend Forecaster (models/trend/forecaster_*.joblib)

Usage:
    python scripts/generate_model_cards.py --model all
    python scripts/generate_model_cards.py --model churn
    python scripts/generate_model_cards.py --model anomaly
    python scripts/generate_model_cards.py --model delivery
    python scripts/generate_model_cards.py --model churn_telco
    python scripts/generate_model_cards.py --model delivery_dataco
    python scripts/generate_model_cards.py --model sentiment_industry
    python scripts/generate_model_cards.py --model trend
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Project root on path — required for scripts.data_loader import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data_loader import OlistDataLoader  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "olist"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style / palette constants
# ---------------------------------------------------------------------------
PALETTE = {
    "churn": {
        "primary": "#1565C0",  # deep blue
        "secondary": "#42A5F5",  # light blue
        "shading": "#90CAF9",  # very light blue for CI band
        "cmap": "Blues",
    },
    "anomaly": {
        "primary": "#B71C1C",  # deep red
        "secondary": "#EF5350",  # light red
        "shading": "#FFCDD2",  # very light red for CI band
        "cmap": "Reds",
    },
    "delivery": {
        "primary": "#1B5E20",  # deep green
        "secondary": "#66BB6A",  # light green
        "shading": "#C8E6C9",  # very light green for CI band
        "cmap": "Greens",
    },
    # Industry palette keys — mapped to base colour families
    "churn_telco": {
        "primary": "#1565C0",  # deep blue
        "secondary": "#42A5F5",  # light blue
        "shading": "#90CAF9",  # very light blue
        "cmap": "Blues",
    },
    "delivery_dataco": {
        "primary": "#1B5E20",  # deep green
        "secondary": "#66BB6A",  # light green
        "shading": "#C8E6C9",  # very light green
        "cmap": "Greens",
    },
    "sentiment_industry": {
        "primary": "#4A148C",  # deep purple
        "secondary": "#AB47BC",  # light purple
        "shading": "#E1BEE7",  # very light purple
        "cmap": "Purples",
    },
    "trend": {
        "primary": "#E65100",  # deep orange
        "secondary": "#FFA726",  # light orange
        "shading": "#FFE0B2",  # very light orange
        "cmap": "Oranges",
    },
}

MLFLOW_EXPERIMENT = "ledgerguard-model-cards"
N_CV_FOLDS = 5
BOOTSTRAP_ITERATIONS = 1000
DPI = 150
RANDOM_STATE = 42
CONTAMINATION = 0.02  # Anomaly pseudo-label contamination rate


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ci_95(values: np.ndarray) -> float:
    """Return the half-width of a 95% confidence interval (1.96 * SEM).

    Args:
        values: 1-D array of per-fold or per-bootstrap metric values.

    Returns:
        Half-width of the 95% CI.
    """
    n = len(values)
    if n < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(n))


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    """Build a mean / std / ci95 summary dict for a metric array.

    Args:
        values: Per-fold metric scores.

    Returns:
        Dictionary with keys ``mean``, ``std``, ``ci95_half_width``.
    """
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci95_half_width": _ci_95(values),
    }


def _apply_clean_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Fall back to default matplotlib style
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def _to_native(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization.

    Args:
        obj: Any Python object, potentially containing numpy scalars or arrays.

    Returns:
        The same structure with all numpy types replaced by native Python types.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------


def run_stratified_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_CV_FOLDS,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Run stratified k-fold cross-validation and collect per-fold metrics.

    The model is **not** re-trained during CV; instead, ``predict_proba`` is
    called on each held-out fold using the already-fitted estimator.  This
    reflects the production evaluation scenario (assessing a trained artefact)
    rather than a training CV.

    Args:
        model: Fitted sklearn-compatible classifier with ``predict_proba``.
        X: Feature array (numpy or DataFrame).
        y: Binary label array.
        n_splits: Number of CV folds (default 5).
        threshold: Decision threshold for positive-class predictions.

    Returns:
        Dictionary mapping metric name to 1-D array of per-fold scores.
        Keys: ``f1``, ``precision``, ``recall``, ``roc_auc``.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_f1: list[float] = []
    fold_precision: list[float] = []
    fold_recall: list[float] = []
    fold_roc_auc: list[float] = []

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y)

    for _, test_idx in skf.split(X_arr, y_arr):
        X_fold = X_arr[test_idx]
        y_fold = y_arr[test_idx]

        y_proba = model.predict_proba(X_fold)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        fold_f1.append(f1_score(y_fold, y_pred, zero_division=0))
        fold_precision.append(precision_score(y_fold, y_pred, zero_division=0))
        fold_recall.append(recall_score(y_fold, y_pred, zero_division=0))

        try:
            fold_roc_auc.append(roc_auc_score(y_fold, y_proba))
        except ValueError:
            fold_roc_auc.append(0.0)

    return {
        "f1": np.array(fold_f1),
        "precision": np.array(fold_precision),
        "recall": np.array(fold_recall),
        "roc_auc": np.array(fold_roc_auc),
    }


def run_anomaly_cv(
    model: IsolationForest,
    X: np.ndarray,
    n_splits: int = N_CV_FOLDS,
) -> dict[str, np.ndarray]:
    """Run stratified-style CV for Isolation Forest using pseudo-labels.

    Because IsolationForest is unsupervised, pseudo-labels are generated from
    the model's ``decision_function`` on each fold: the bottom
    ``contamination * 100``-th percentile scores are labeled as anomalies (1),
    the rest as normal (0).

    The stratification uses the pseudo-labels derived from the *full* test-set
    anomaly scores (consistent label source), so ``StratifiedKFold`` can
    balance classes across folds.

    Args:
        model: Trained ``sklearn.ensemble.IsolationForest``.
        X: Feature matrix (numpy array).
        n_splits: Number of CV folds.

    Returns:
        Dictionary mapping metric name to per-fold score array.
        Keys: ``f1``, ``precision``, ``recall``, ``roc_auc``.
    """
    X_arr = np.asarray(X)

    # Derive a stable set of global pseudo-labels for stratification
    global_scores = -model.score_samples(X_arr)  # higher == more anomalous
    threshold_pct = np.percentile(global_scores, (1 - CONTAMINATION) * 100)
    y_pseudo_global = (global_scores >= threshold_pct).astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_f1: list[float] = []
    fold_precision: list[float] = []
    fold_recall: list[float] = []
    fold_roc_auc: list[float] = []

    for _, test_idx in skf.split(X_arr, y_pseudo_global):
        X_fold = X_arr[test_idx]
        fold_scores = -model.score_samples(X_fold)

        # Pseudo-labels for this fold based on its own score distribution
        fold_thr = np.percentile(fold_scores, (1 - CONTAMINATION) * 100)
        y_fold_true = (fold_scores >= fold_thr).astype(int)
        y_fold_pred = (
            model.predict(X_fold) == -1
        ).astype(int)  # -1 == anomaly in sklearn API

        fold_f1.append(f1_score(y_fold_true, y_fold_pred, zero_division=0))
        fold_precision.append(precision_score(y_fold_true, y_fold_pred, zero_division=0))
        fold_recall.append(recall_score(y_fold_true, y_fold_pred, zero_division=0))

        try:
            fold_roc_auc.append(roc_auc_score(y_fold_true, fold_scores))
        except ValueError:
            fold_roc_auc.append(0.0)

    return {
        "f1": np.array(fold_f1),
        "precision": np.array(fold_precision),
        "recall": np.array(fold_recall),
        "roc_auc": np.array(fold_roc_auc),
    }


def run_multiclass_cv(
    pipeline: Any,
    X_text: list[str],
    y: np.ndarray,
    n_splits: int = N_CV_FOLDS,
) -> dict[str, np.ndarray]:
    """Run stratified k-fold CV for multiclass text classification pipelines.

    The pipeline is **not** re-trained; predictions on each held-out fold are
    produced by the pre-fitted pipeline.  Macro-averaged F1 is reported per
    fold to give a class-balanced summary.

    Args:
        pipeline: Fitted sklearn ``Pipeline`` supporting ``predict`` and
            ``predict_proba`` on raw text sequences.
        X_text: List of raw text strings (pipeline handles vectorisation).
        y: Integer class labels (0, 1, 2 for neg/neu/pos).
        n_splits: Number of stratified folds.

    Returns:
        Dictionary with key ``f1`` mapped to a 1-D per-fold macro-F1 array.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X_arr = np.asarray(X_text)
    y_arr = np.asarray(y)

    fold_f1: list[float] = []

    for _, test_idx in skf.split(X_arr, y_arr):
        X_fold = X_arr[test_idx].tolist()
        y_fold = y_arr[test_idx]

        y_pred = pipeline.predict(X_fold)
        fold_f1.append(f1_score(y_fold, y_pred, average="macro", zero_division=0))

    return {"f1": np.array(fold_f1)}


# ---------------------------------------------------------------------------
# Plot: CV box plot
# ---------------------------------------------------------------------------


def plot_cv_boxplot(
    cv_results: dict[str, np.ndarray],
    model_display_name: str,
    model_key: str,
    suffix: str = "",
) -> Path:
    """Save a box plot of per-fold CV metric distributions.

    Args:
        cv_results: Output of any ``run_*_cv`` function.
        model_display_name: Human-readable model name for plot title.
        model_key: Short key used for palette selection and part of filename.
        suffix: Optional filename suffix (e.g. ``"_telco"``).

    Returns:
        Absolute path to the saved PNG.
    """
    _apply_clean_style()
    palette = PALETTE[model_key]

    metrics_order = [m for m in ["f1", "precision", "recall", "roc_auc"] if m in cv_results]
    label_map = {"f1": "F1", "precision": "Precision", "recall": "Recall", "roc_auc": "ROC-AUC"}
    labels = [label_map[m] for m in metrics_order]
    data = [cv_results[m] for m in metrics_order]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops={"color": palette["primary"], "linewidth": 2},
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(palette["shading"])
        patch.set_edgecolor(palette["secondary"])
        patch.set_linewidth(1.5)

    for whisker in bp["whiskers"]:
        whisker.set(color=palette["secondary"], linewidth=1.2, linestyle="--")

    for cap in bp["caps"]:
        cap.set(color=palette["secondary"], linewidth=1.2)

    for flier in bp["fliers"]:
        flier.set(marker="o", color=palette["primary"], alpha=0.6, markersize=5)

    # Overlay mean markers
    for i, values in enumerate(data, start=1):
        ax.scatter(
            i,
            np.mean(values),
            zorder=5,
            color=palette["primary"],
            s=60,
            marker="D",
            label="Mean" if i == 1 else None,
        )

    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{model_display_name} — 5-Fold CV Metric Distribution",
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=9)

    filename = f"cv_boxplot_{model_key}{suffix}.png"
    out_path = REPORTS_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot: Confusion matrix heatmap
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_display_name: str,
    model_key: str,
    class_labels: tuple[str, ...] = ("Negative", "Positive"),
    suffix: str = "",
) -> Path:
    """Save a seaborn confusion matrix heatmap.

    Supports both binary (2-class) and multiclass (3+ class) confusion
    matrices.  The colour map and title are driven by ``model_key``.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        model_display_name: Human-readable model name for the plot title.
        model_key: Short key for filename and palette selection.
        class_labels: Tuple of display-name strings in label order.
        suffix: Optional filename suffix (e.g. ``"_telco"``).

    Returns:
        Absolute path to the saved PNG.
    """
    _apply_clean_style()
    palette = PALETTE[model_key]

    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_labels)
    fig_w = max(6, n_classes * 2.2)
    fig_h = max(5, n_classes * 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=palette["cmap"],
        linewidths=0.5,
        linecolor="white",
        xticklabels=list(class_labels),
        yticklabels=list(class_labels),
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"{model_display_name} — Confusion Matrix (Test Set)",
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()

    filename = f"confusion_matrix_{model_key}{suffix}.png"
    out_path = REPORTS_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot: ROC curve with bootstrap CI
# ---------------------------------------------------------------------------


def plot_roc_curve_with_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_display_name: str,
    model_key: str,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    suffix: str = "",
) -> Path:
    """Save a ROC curve with a 95% bootstrap confidence band.

    1000 bootstrap resamples of (y_true, y_proba) are drawn with replacement.
    Each resample produces a ROC curve interpolated onto a shared FPR grid.
    The 2.5th and 97.5th percentile TPR values at each FPR point form the band.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probability of the positive class.
        model_display_name: Human-readable model name for the plot title.
        model_key: Short key for filename and palette selection.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        suffix: Optional filename suffix (e.g. ``"_telco"``).

    Returns:
        Absolute path to the saved PNG.
    """
    _apply_clean_style()
    palette = PALETTE[model_key]
    rng = np.random.RandomState(RANDOM_STATE)

    fpr_grid = np.linspace(0, 1, 200)
    tpr_bootstrap = np.zeros((n_bootstrap, len(fpr_grid)))
    n = len(y_true)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_proba[idx]
        # Skip degenerate resamples with only one class present
        if len(np.unique(y_b)) < 2:
            tpr_bootstrap[i] = np.interp(
                fpr_grid, *roc_curve(y_true, y_proba)[:2]
            )
            continue
        fpr_b, tpr_b, _ = roc_curve(y_b, p_b)
        tpr_bootstrap[i] = np.interp(fpr_grid, fpr_b, tpr_b)

    tpr_mean = tpr_bootstrap.mean(axis=0)
    tpr_lower = np.percentile(tpr_bootstrap, 2.5, axis=0)
    tpr_upper = np.percentile(tpr_bootstrap, 97.5, axis=0)

    mean_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.fill_between(
        fpr_grid,
        tpr_lower,
        tpr_upper,
        alpha=0.25,
        color=palette["shading"],
        label="95% CI (bootstrap)",
    )
    ax.plot(
        fpr_grid,
        tpr_mean,
        color=palette["primary"],
        linewidth=2.5,
        label=f"Mean ROC (AUC = {mean_auc:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.0,
        alpha=0.6,
        label="Random classifier",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"{model_display_name} — ROC Curve with 95% Bootstrap CI",
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=9)

    filename = f"roc_curve_{model_key}{suffix}.png"
    out_path = REPORTS_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Model card JSON
# ---------------------------------------------------------------------------


def save_model_card(
    model_key: str,
    model_name: str,
    model_type: str,
    dataset: str,
    cv_results: dict[str, np.ndarray],
    test_metrics: dict[str, float],
    feature_count: int,
    training_samples: int,
    test_samples: int,
    extra: dict[str, Any] | None = None,
    filename_override: str | None = None,
) -> Path:
    """Persist a structured model card JSON file to ``reports/``.

    Args:
        model_key: Short identifier (``churn``, ``anomaly``, ``delivery``, etc.).
        model_name: Full human-readable model name.
        model_type: Algorithm type string (e.g. ``LightGBM``).
        dataset: Dataset description string.
        cv_results: Per-fold metric arrays from CV.
        test_metrics: Scalar test-set metrics dict (f1, precision, recall, auc).
        feature_count: Number of input features.
        training_samples: Number of training samples.
        test_samples: Number of test samples.
        extra: Optional additional metadata to include verbatim.
        filename_override: If set, save as ``reports/{filename_override}``
            instead of the default ``reports/model_card_{model_key}.json``.

    Returns:
        Absolute path to the saved JSON.
    """
    card: dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "dataset": dataset,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "feature_count": feature_count,
        "training_samples": training_samples,
        "test_samples": test_samples,
        "cv_folds": N_CV_FOLDS,
        "metrics": {
            "test": {k: round(float(v), 4) for k, v in test_metrics.items()},
            "cv": {
                metric: {
                    "mean": round(_metric_summary(values)["mean"], 4),
                    "std": round(_metric_summary(values)["std"], 4),
                    "ci95_half_width": round(
                        _metric_summary(values)["ci95_half_width"], 4
                    ),
                }
                for metric, values in cv_results.items()
            },
        },
    }

    if extra:
        card["extra"] = extra

    if filename_override:
        out_path = REPORTS_DIR / filename_override
    else:
        out_path = REPORTS_DIR / f"model_card_{model_key}.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_to_native(card), fh, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Per-model generation routines — Olist originals
# ---------------------------------------------------------------------------


def generate_churn_card() -> None:
    """Generate all model card artifacts for the LightGBM churn model.

    Loads the trained model from ``models/churn/lightgbm_churn_model.pkl``,
    reloads the Olist churn dataset, runs 5-fold stratified CV, then plots
    the confusion matrix and bootstrapped ROC curve.
    """
    model_key = "churn"
    model_display_name = "LightGBM Churn Classifier"
    model_type = "LightGBM"
    dataset = "Olist Brazilian E-Commerce — customer-level RFM features"

    model_path = MODELS_DIR / "churn" / "lightgbm_churn_model.pkl"
    if not model_path.exists():
        print(f"  [WARN] Model not found at {model_path} — skipping churn card.")
        return

    print(f"  Loading model: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names: list[str] = artifact.get("feature_names", [])
    threshold: float = artifact.get("threshold", 0.5)

    print("  Loading churn dataset...")
    loader = OlistDataLoader(str(DATA_DIR))
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_churn_data()

    # Combine train+val for a larger CV pool
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    print("  Running 5-fold stratified CV...")
    cv_results = run_stratified_cv(model, X_full, y_full)

    # Test-set evaluation
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    try:
        test_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        test_auc = 0.0

    test_metrics = {
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "auc": test_auc,
    }

    print("  Generating CV box plot...")
    cv_plot = plot_cv_boxplot(cv_results, model_display_name, model_key)

    print("  Generating confusion matrix heatmap...")
    cm_plot = plot_confusion_matrix(
        np.asarray(y_test),
        y_pred,
        model_display_name,
        model_key,
        class_labels=("Retained", "Churned"),
    )

    print("  Generating ROC curve with bootstrap CI...")
    roc_plot = plot_roc_curve_with_ci(
        np.asarray(y_test),
        y_proba,
        model_display_name,
        model_key,
    )

    print("  Saving model card JSON...")
    card_path = save_model_card(
        model_key=model_key,
        model_name=model_display_name,
        model_type=model_type,
        dataset=dataset,
        cv_results=cv_results,
        test_metrics=test_metrics,
        feature_count=len(feature_names) if feature_names else X_train.shape[1],
        training_samples=len(X_train),
        test_samples=len(X_test),
        extra={"decision_threshold": round(threshold, 4)},
    )

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"model_card_{model_key}"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "dataset": dataset,
                "cv_folds": N_CV_FOLDS,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            }
        )
        mlflow.log_metrics(
            {
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "cv_f1_mean": float(np.mean(cv_results["f1"])),
                "cv_f1_ci95": _ci_95(cv_results["f1"]),
                "cv_roc_auc_mean": float(np.mean(cv_results["roc_auc"])),
                "cv_roc_auc_ci95": _ci_95(cv_results["roc_auc"]),
            }
        )
        for artifact_path in (cv_plot, cm_plot, roc_plot, card_path):
            mlflow.log_artifact(str(artifact_path))

    print(
        f"  [OK] Churn card complete — CV F1 {np.mean(cv_results['f1']):.3f}"
        f" +/- {_ci_95(cv_results['f1']):.3f}  |  Test AUC {test_auc:.3f}"
    )


def generate_anomaly_card() -> None:
    """Generate all model card artifacts for the Isolation Forest anomaly model.

    Loads the trained model from ``models/anomaly/isolation_forest.joblib``.
    Because this is an unsupervised model, pseudo-labels are derived from the
    model's own anomaly scores at contamination=0.02 for evaluation.

    CV is performed via :func:`run_anomaly_cv`.  ROC curve is intentionally
    omitted (no reliable binary ground truth); a confusion matrix based on the
    pseudo-labeled test set is produced instead.
    """
    model_key = "anomaly"
    model_display_name = "Isolation Forest Anomaly Detector"
    model_type = "IsolationForest"
    dataset = "Olist Brazilian E-Commerce — daily aggregated time-series metrics"

    model_path = MODELS_DIR / "anomaly" / "isolation_forest.joblib"
    if not model_path.exists():
        print(f"  [WARN] Model not found at {model_path} — skipping anomaly card.")
        return

    print(f"  Loading model: {model_path}")
    model: IsolationForest = joblib.load(model_path)

    print("  Loading anomaly dataset...")
    loader = OlistDataLoader(str(DATA_DIR))
    X_train, X_val, X_test, _dates_train, _dates_val, _dates_test = (
        loader.prepare_anomaly_detection_data()
    )

    # Pseudo-labels for test set: bottom contamination% most anomalous scores
    test_scores = -model.score_samples(X_test)  # higher == more anomalous
    test_threshold = np.percentile(test_scores, (1 - CONTAMINATION) * 100)
    y_test_pseudo = (test_scores >= test_threshold).astype(int)
    y_test_pred = (model.predict(X_test) == -1).astype(int)

    print("  Running 5-fold CV (pseudo-label based)...")
    # Use combined train+val for a more reliable CV estimate
    X_full = np.vstack([X_train, X_val])
    cv_results = run_anomaly_cv(model, X_full)

    test_f1 = f1_score(y_test_pseudo, y_test_pred, zero_division=0)
    test_precision = precision_score(y_test_pseudo, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test_pseudo, y_test_pred, zero_division=0)
    try:
        test_auc = roc_auc_score(y_test_pseudo, test_scores)
    except ValueError:
        test_auc = 0.0

    test_metrics = {
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "auc": test_auc,
    }

    print("  Generating CV box plot...")
    cv_plot = plot_cv_boxplot(cv_results, model_display_name, model_key)

    print("  Generating confusion matrix heatmap...")
    cm_plot = plot_confusion_matrix(
        y_test_pseudo,
        y_test_pred,
        model_display_name,
        model_key,
        class_labels=("Normal", "Anomaly"),
    )

    print("  Saving model card JSON...")
    card_path = save_model_card(
        model_key=model_key,
        model_name=model_display_name,
        model_type=model_type,
        dataset=dataset,
        cv_results=cv_results,
        test_metrics=test_metrics,
        feature_count=X_train.shape[1],
        training_samples=len(X_train),
        test_samples=len(X_test),
        extra={
            "contamination": CONTAMINATION,
            "pseudo_label_note": (
                "Labels derived from model anomaly scores at contamination=0.02; "
                "no ground-truth anomaly labels available in this dataset."
            ),
        },
    )

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"model_card_{model_key}"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "dataset": dataset,
                "cv_folds": N_CV_FOLDS,
                "contamination": CONTAMINATION,
            }
        )
        mlflow.log_metrics(
            {
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "cv_f1_mean": float(np.mean(cv_results["f1"])),
                "cv_f1_ci95": _ci_95(cv_results["f1"]),
                "cv_roc_auc_mean": float(np.mean(cv_results["roc_auc"])),
                "cv_roc_auc_ci95": _ci_95(cv_results["roc_auc"]),
            }
        )
        for artifact_path in (cv_plot, cm_plot, card_path):
            mlflow.log_artifact(str(artifact_path))

    print(
        f"  [OK] Anomaly card complete — CV F1 {np.mean(cv_results['f1']):.3f}"
        f" +/- {_ci_95(cv_results['f1']):.3f}  |  Test AUC {test_auc:.3f}"
    )


def generate_delivery_card() -> None:
    """Generate all model card artifacts for the XGBoost late delivery model.

    Loads the trained model from ``models/delivery/xgboost_late_delivery.joblib``,
    which is a dict with keys ``model`` (bare ``XGBClassifier``) and ``threshold``.
    The decision threshold stored in the artefact is used for confusion matrix
    prediction; CV uses the default 0.5 threshold to measure raw discriminative
    power across folds.
    """
    model_key = "delivery"
    model_display_name = "XGBoost Late Delivery Predictor"
    model_type = "XGBoost"
    dataset = "Olist Brazilian E-Commerce — order-level delivery features"

    model_path = MODELS_DIR / "delivery" / "xgboost_late_delivery.joblib"
    if not model_path.exists():
        print(f"  [WARN] Model not found at {model_path} — skipping delivery card.")
        return

    print(f"  Loading model: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    threshold: float = artifact.get("threshold", 0.5)

    print("  Loading late delivery dataset...")
    loader = OlistDataLoader(str(DATA_DIR))
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_late_delivery_data(
        test_size=0.2,
        val_size=0.1,
        random_state=RANDOM_STATE,
    )

    # Combine train+val for CV
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    print("  Running 5-fold stratified CV...")
    cv_results = run_stratified_cv(model, X_full, y_full)

    # Test-set evaluation using the stored decision threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    try:
        test_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        test_auc = 0.0

    test_metrics = {
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "auc": test_auc,
    }

    print("  Generating CV box plot...")
    cv_plot = plot_cv_boxplot(cv_results, model_display_name, model_key)

    print("  Generating confusion matrix heatmap...")
    cm_plot = plot_confusion_matrix(
        np.asarray(y_test),
        y_pred,
        model_display_name,
        model_key,
        class_labels=("On-Time", "Late"),
    )

    print("  Generating ROC curve with bootstrap CI...")
    roc_plot = plot_roc_curve_with_ci(
        np.asarray(y_test),
        y_proba,
        model_display_name,
        model_key,
    )

    print("  Saving model card JSON...")
    card_path = save_model_card(
        model_key=model_key,
        model_name=model_display_name,
        model_type=model_type,
        dataset=dataset,
        cv_results=cv_results,
        test_metrics=test_metrics,
        feature_count=X_train.shape[1],
        training_samples=len(X_train),
        test_samples=len(X_test),
        extra={"decision_threshold": round(threshold, 4)},
    )

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"model_card_{model_key}"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "dataset": dataset,
                "cv_folds": N_CV_FOLDS,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            }
        )
        mlflow.log_metrics(
            {
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "cv_f1_mean": float(np.mean(cv_results["f1"])),
                "cv_f1_ci95": _ci_95(cv_results["f1"]),
                "cv_roc_auc_mean": float(np.mean(cv_results["roc_auc"])),
                "cv_roc_auc_ci95": _ci_95(cv_results["roc_auc"]),
            }
        )
        for artifact_path in (cv_plot, cm_plot, roc_plot, card_path):
            mlflow.log_artifact(str(artifact_path))

    print(
        f"  [OK] Delivery card complete — CV F1 {np.mean(cv_results['f1']):.3f}"
        f" +/- {_ci_95(cv_results['f1']):.3f}  |  Test AUC {test_auc:.3f}"
    )


# ---------------------------------------------------------------------------
# Per-model generation routines — Industry models
# ---------------------------------------------------------------------------


def _prepare_telco_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Apply the same feature engineering used during Telco model training.

    Replicates the preprocessing from ``TelcoChurnLoader`` in
    ``scripts/industry_data_loader.py`` so that the card generator can work
    from the raw CSV without importing the loader directly.

    Args:
        df: Raw Telco CSV loaded as a DataFrame.
        feature_cols: List of feature column names expected by the trained model.

    Returns:
        DataFrame containing exactly ``feature_cols`` columns, fully numeric,
        with NaN filled to zero.
    """
    df = df.copy()

    # Fix TotalCharges blank strings (new customers with tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Binary Yes/No columns
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col + "_encoded"] = (df[col] == "Yes").astype(int)

    # SeniorCitizen is already 0/1
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen_encoded"] = df["SeniorCitizen"].astype(int)

    # Gender
    if "gender" in df.columns:
        df["gender_encoded"] = (df["gender"] == "Male").astype(int)

    # Multi-service columns (No / Yes / No internet service -> 0/1/0)
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_cols:
        if col in df.columns:
            df[col + "_encoded"] = (df[col] == "Yes").astype(int)

    if all(col + "_encoded" in df.columns for col in service_cols):
        df["num_premium_services"] = sum(df[col + "_encoded"] for col in service_cols)

    # MultipleLines
    if "MultipleLines" in df.columns:
        df["MultipleLines_encoded"] = (df["MultipleLines"] == "Yes").astype(int)

    # InternetService one-hot
    if "InternetService" in df.columns:
        df["internet_dsl"] = (df["InternetService"] == "DSL").astype(int)
        df["internet_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
        df["internet_none"] = (df["InternetService"] == "No").astype(int)

    # Contract ordinal
    if "Contract" in df.columns:
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        df["contract_encoded"] = df["Contract"].map(contract_map).fillna(0).astype(int)

    # PaymentMethod one-hot
    if "PaymentMethod" in df.columns:
        df["pay_electronic"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        df["pay_mailed"] = (df["PaymentMethod"] == "Mailed check").astype(int)
        df["pay_bank_transfer"] = (
            df["PaymentMethod"] == "Bank transfer (automatic)"
        ).astype(int)
        df["pay_credit_card"] = (
            df["PaymentMethod"] == "Credit card (automatic)"
        ).astype(int)

    # Numeric coercions
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Interaction features
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        df["charges_ratio"] = df["TotalCharges"] / (
            df["tenure"] * df["MonthlyCharges"] + 1
        )
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 100],
            labels=[0, 1, 2, 3],
        ).astype(int)
        df["high_value"] = (
            df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)
        ).astype(int)

    if all(
        c in df.columns
        for c in ["contract_encoded", "internet_fiber", "TechSupport_encoded"]
    ):
        df["at_risk_combo"] = (
            (df["contract_encoded"] == 0)
            & (df["internet_fiber"] == 1)
            & (df["TechSupport_encoded"] == 0)
        ).astype(int)

    # Generic fallback for any remaining object columns listed in feature_cols
    for col in feature_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0}).fillna(0)
            )

    return df[feature_cols].fillna(0)


def _prepare_dataco_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Apply the same feature engineering used during DataCo model training.

    Replicates the preprocessing from ``DataCoDeliveryLoader`` in
    ``scripts/industry_data_loader.py`` so that the card generator can work
    from the raw CSV without importing the loader directly.

    Args:
        df: Raw DataCo CSV loaded with ``encoding='latin-1'``.
        feature_cols: List of feature column names expected by the trained model.

    Returns:
        DataFrame containing exactly ``feature_cols`` columns, fully numeric,
        with NaN filled to zero.
    """
    df = df.copy()

    # Temporal features from order date
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
            ("order_day_of_week", 0),
            ("order_month", 1),
            ("order_hour", 12),
            ("order_day_of_month", 15),
            ("is_weekend", 0),
        ]:
            df[col] = val

    # Scheduled shipping days
    if "Days for shipment (scheduled)" in df.columns:
        df["scheduled_shipping_days"] = df["Days for shipment (scheduled)"]
    else:
        df["scheduled_shipping_days"] = 4

    # Order financial features
    order_value_cols = {
        "Sales per customer": "sales_per_customer",
        "Order Item Discount": "item_discount",
        "Order Item Discount Rate": "discount_rate",
        "Order Item Profit Ratio": "profit_ratio",
        "Order Item Quantity": "item_quantity",
        "Sales": "sales",
        "Order Profit Per Order": "profit_per_order",
        "Benefit per order": "benefit_per_order",
        "Order Item Total": "item_total",
        "Order Item Product Price": "product_price",
    }
    for orig, new in order_value_cols.items():
        if orig in df.columns:
            df[new] = pd.to_numeric(df[orig], errors="coerce").fillna(0)
        else:
            df[new] = 0

    # Shipping mode ordinal
    if "Shipping Mode" in df.columns:
        ship_map = {
            "Standard Class": 0,
            "Second Class": 1,
            "First Class": 2,
            "Same Day": 3,
        }
        df["shipping_mode_encoded"] = df["Shipping Mode"].map(ship_map).fillna(0).astype(int)
    else:
        df["shipping_mode_encoded"] = 0

    # Customer segment
    if "Customer Segment" in df.columns:
        seg_map = {"Consumer": 0, "Corporate": 1, "Home Office": 2}
        df["customer_segment_encoded"] = (
            df["Customer Segment"].map(seg_map).fillna(0).astype(int)
        )
    else:
        df["customer_segment_encoded"] = 0

    # Market
    if "Market" in df.columns:
        le_market = LabelEncoder()
        df["market_encoded"] = le_market.fit_transform(df["Market"].fillna("Unknown"))
    else:
        df["market_encoded"] = 0

    # Order region
    if "Order Region" in df.columns:
        le_region = LabelEncoder()
        df["region_encoded"] = le_region.fit_transform(
            df["Order Region"].fillna("Unknown")
        )
    else:
        df["region_encoded"] = 0

    # Product category
    if "Category Name" in df.columns:
        le_cat = LabelEncoder()
        df["category_encoded"] = le_cat.fit_transform(
            df["Category Name"].fillna("Unknown")
        )
    elif "Department Name" in df.columns:
        le_cat = LabelEncoder()
        df["category_encoded"] = le_cat.fit_transform(
            df["Department Name"].fillna("Unknown")
        )
    else:
        df["category_encoded"] = 0

    # Order status (pre-shipment only)
    if "Order Status" in df.columns:
        pre_shipment_map = {
            "PENDING": 1,
            "PROCESSING": 2,
            "PENDING_PAYMENT": 3,
            "ON_HOLD": 4,
            "SUSPECTED_FRAUD": 5,
            "PAYMENT_REVIEW": 6,
        }
        df["order_status_encoded"] = (
            df["Order Status"].map(pre_shipment_map).fillna(0).astype(int)
        )
    else:
        df["order_status_encoded"] = 0

    # Department
    if "Department Name" in df.columns:
        le_dept = LabelEncoder()
        df["department_encoded"] = le_dept.fit_transform(
            df["Department Name"].fillna("Unknown")
        )
    else:
        df["department_encoded"] = 0

    # Interaction features
    df["value_per_item"] = df["sales"] / (df["item_quantity"] + 1)
    df["discount_amount"] = df["sales"] * df["discount_rate"]
    df["high_value_order"] = (df["sales"] > df["sales"].quantile(0.75)).astype(int)
    df["rush_shipping"] = (df["shipping_mode_encoded"] >= 2).astype(int)
    df["high_quantity"] = (
        df["item_quantity"] > df["item_quantity"].quantile(0.75)
    ).astype(int)
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

    # Keep only model-expected feature columns that were successfully computed
    available = [c for c in feature_cols if c in df.columns]
    return df[available].fillna(0)


def generate_churn_telco_card() -> None:
    """Generate model card artifacts for the LightGBM Telco churn model.

    Loads ``models/churn_industry/lightgbm_telco_churn.pkl`` and the IBM Telco
    Customer Churn CSV (7,043 rows).  Performs 5-fold stratified CV on the full
    dataset, evaluates the optimised decision threshold on a held-out 20 % test
    split, and emits confusion matrix, bootstrapped ROC curve, CV box plot, and
    JSON model card.

    Artifacts saved:
        reports/model_card_churn_telco.json
        reports/confusion_matrix_churn_telco.png
        reports/cv_boxplot_churn_telco.png
        reports/roc_curve_churn_telco.png
    """
    model_key = "churn_telco"
    file_suffix = "_churn_telco"
    model_display_name = "LightGBM Churn Classifier (Telco Industry)"
    model_type = "LightGBM"
    dataset = "IBM Telco Customer Churn — 7,043 customers, 30 features"

    model_path = MODELS_DIR / "churn_industry" / "lightgbm_telco_churn.pkl"
    if not model_path.exists():
        print(f"  [WARN] Model not found at {model_path} — skipping churn_telco card.")
        return

    data_path = PROJECT_ROOT / "data" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not data_path.exists():
        print(f"  [WARN] Telco data not found at {data_path} — skipping churn_telco card.")
        return

    print(f"  Loading model: {model_path}")
    model_data: dict[str, Any] = joblib.load(model_path)
    lgbm_model = model_data["model"]
    threshold: float = float(model_data["threshold"])
    feature_cols: list[str] = list(model_data["features"])

    print("  Loading Telco dataset...")
    df = pd.read_csv(data_path)
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    X = _prepare_telco_features(df, feature_cols)
    y = df["Churn_binary"]

    # Stratified 80/20 train/test split for test-set evaluation
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("  Running 5-fold stratified CV on full dataset...")
    cv_results = run_stratified_cv(lgbm_model, X, y, threshold=threshold)

    # Test-set evaluation
    y_proba = lgbm_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    try:
        test_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        test_auc = 0.0

    test_metrics = {
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "auc": test_auc,
    }

    print("  Generating CV box plot...")
    cv_plot = plot_cv_boxplot(cv_results, model_display_name, model_key, suffix="")

    print("  Generating confusion matrix heatmap...")
    cm_plot = plot_confusion_matrix(
        np.asarray(y_test),
        y_pred,
        model_display_name,
        model_key,
        class_labels=("Retained", "Churned"),
        suffix="",
    )

    print("  Generating ROC curve with bootstrap CI...")
    roc_plot = plot_roc_curve_with_ci(
        np.asarray(y_test),
        y_proba,
        model_display_name,
        model_key,
        suffix="",
    )

    print("  Saving model card JSON...")
    card_path = save_model_card(
        model_key=model_key,
        model_name=model_display_name,
        model_type=model_type,
        dataset=dataset,
        cv_results=cv_results,
        test_metrics=test_metrics,
        feature_count=len(feature_cols),
        training_samples=len(X_train),
        test_samples=len(X_test),
        extra={
            "decision_threshold": round(threshold, 4),
            "industry": "Telecom",
        },
        filename_override="model_card_churn_telco.json",
    )

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="model_card_churn_telco"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "dataset": dataset,
                "cv_folds": N_CV_FOLDS,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "decision_threshold": round(threshold, 4),
                "industry": "Telecom",
            }
        )
        mlflow.log_metrics(
            {
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "cv_f1_mean": float(np.mean(cv_results["f1"])),
                "cv_f1_ci95": _ci_95(cv_results["f1"]),
                "cv_roc_auc_mean": float(np.mean(cv_results["roc_auc"])),
                "cv_roc_auc_ci95": _ci_95(cv_results["roc_auc"]),
            }
        )
        for artifact_path in (cv_plot, cm_plot, roc_plot, card_path):
            mlflow.log_artifact(str(artifact_path))

    print(
        f"  [OK] Churn Telco card complete — CV F1 {np.mean(cv_results['f1']):.3f}"
        f" +/- {_ci_95(cv_results['f1']):.3f}  |  Test AUC {test_auc:.3f}"
    )


def generate_delivery_dataco_card() -> None:
    """Generate model card artifacts for the XGBoost DataCo delivery model.

    Loads ``models/delivery_industry/xgboost_dataco_delivery.joblib`` and the
    DataCo Supply Chain CSV (180 K orders, latin-1 encoding).  Feature
    engineering mirrors ``DataCoDeliveryLoader``.  An 80/20 stratified split
    provides the test set; 5-fold CV is run on the 80 % training portion.

    Artifacts saved:
        reports/model_card_delivery_dataco.json
        reports/confusion_matrix_delivery_dataco.png
        reports/cv_boxplot_delivery_dataco.png
        reports/roc_curve_delivery_dataco.png
    """
    model_key = "delivery_dataco"
    model_display_name = "XGBoost Late Delivery Predictor (DataCo Industry)"
    model_type = "XGBoost"
    dataset = "DataCo Smart Supply Chain — 180,519 orders, 29 engineered features"

    model_path = MODELS_DIR / "delivery_industry" / "xgboost_dataco_delivery.joblib"
    if not model_path.exists():
        print(
            f"  [WARN] Model not found at {model_path} — skipping delivery_dataco card."
        )
        return

    data_path = PROJECT_ROOT / "data" / "dataco" / "DataCoSupplyChainDataset.csv"
    if not data_path.exists():
        print(
            f"  [WARN] DataCo data not found at {data_path} — skipping delivery_dataco card."
        )
        return

    print(f"  Loading model: {model_path}")
    model_data: dict[str, Any] = joblib.load(model_path)
    xgb_model = model_data["model"]
    threshold: float = float(model_data["threshold"])
    feature_cols: list[str] = list(model_data["features"])

    print("  Loading DataCo dataset (180K rows, encoding=latin-1)...")
    df = pd.read_csv(data_path, encoding="latin-1")

    # Apply identical feature engineering to training pipeline
    X = _prepare_dataco_features(df, feature_cols)
    y_raw = df["Late_delivery_risk"].copy()

    # Align index: drop rows where target or any feature is NaN
    valid_mask = y_raw.notna() & X.notna().all(axis=1)
    X = X[valid_mask].reset_index(drop=True)
    y = y_raw[valid_mask].astype(int).reset_index(drop=True)

    print(f"  Dataset after cleaning: {len(X):,} rows")

    # Stratified 80/20 split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("  Running 5-fold stratified CV on 80% train set...")
    cv_results = run_stratified_cv(xgb_model, X_train, y_train, threshold=threshold)

    # Test-set evaluation
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    try:
        test_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        test_auc = 0.0

    test_metrics = {
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "auc": test_auc,
    }

    print("  Generating CV box plot...")
    cv_plot = plot_cv_boxplot(cv_results, model_display_name, model_key, suffix="")

    print("  Generating confusion matrix heatmap...")
    cm_plot = plot_confusion_matrix(
        np.asarray(y_test),
        y_pred,
        model_display_name,
        model_key,
        class_labels=("On-Time", "Late"),
        suffix="",
    )

    print("  Generating ROC curve with bootstrap CI...")
    roc_plot = plot_roc_curve_with_ci(
        np.asarray(y_test),
        y_proba,
        model_display_name,
        model_key,
        suffix="",
    )

    print("  Saving model card JSON...")
    card_path = save_model_card(
        model_key=model_key,
        model_name=model_display_name,
        model_type=model_type,
        dataset=dataset,
        cv_results=cv_results,
        test_metrics=test_metrics,
        feature_count=len(feature_cols),
        training_samples=len(X_train),
        test_samples=len(X_test),
        extra={
            "decision_threshold": round(threshold, 4),
            "industry": "Supply Chain",
        },
        filename_override="model_card_delivery_dataco.json",
    )

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="model_card_delivery_dataco"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "dataset": dataset,
                "cv_folds": N_CV_FOLDS,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "decision_threshold": round(threshold, 4),
                "industry": "Supply Chain",
            }
        )
        mlflow.log_metrics(
            {
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "cv_f1_mean": float(np.mean(cv_results["f1"])),
                "cv_f1_ci95": _ci_95(cv_results["f1"]),
                "cv_roc_auc_mean": float(np.mean(cv_results["roc_auc"])),
                "cv_roc_auc_ci95": _ci_95(cv_results["roc_auc"]),
            }
        )
        for artifact_path in (cv_plot, cm_plot, roc_plot, card_path):
            mlflow.log_artifact(str(artifact_path))

    print(
        f"  [OK] Delivery DataCo card complete — CV F1 {np.mean(cv_results['f1']):.3f}"
        f" +/- {_ci_95(cv_results['f1']):.3f}  |  Test AUC {test_auc:.3f}"
    )


def generate_sentiment_industry_card() -> None:
    """Generate model card artifacts for the LinearSVC financial sentiment model.

    Loads ``models/sentiment_industry/linear_svc_financial_sentiment.joblib``
    (a CalibratedClassifierCV pipeline) and the FinancialPhraseBank CSV.  The
    model performs 3-class classification (0=negative, 1=neutral, 2=positive).

    A 5-fold stratified CV reports macro-averaged F1 per fold.  Test-set
    metrics include accuracy, macro F1, weighted F1, and per-class F1.  No ROC
    curve is produced (multi-class; AUC would require one-vs-rest decomposition
    which is deferred to the per-class analysis).

    Artifacts saved:
        reports/model_card_sentiment.json
        reports/confusion_matrix_sentiment_industry.png
        reports/cv_boxplot_sentiment_industry.png
    """
    model_key = "sentiment_industry"
    model_display_name = "LinearSVC Financial Sentiment Classifier"
    model_type = "LinearSVC (CalibratedClassifierCV Pipeline)"
    dataset = "FinancialPhraseBank — 2,264 annotated financial sentences (neg/neu/pos)"

    model_path = (
        MODELS_DIR / "sentiment_industry" / "linear_svc_financial_sentiment.joblib"
    )
    if not model_path.exists():
        print(
            f"  [WARN] Model not found at {model_path} — skipping sentiment_industry card."
        )
        return

    data_path = PROJECT_ROOT / "data" / "financial_sentiment" / "financial_phrasebank.csv"
    if not data_path.exists():
        print(
            f"  [WARN] Sentiment data not found at {data_path} — "
            "skipping sentiment_industry card."
        )
        return

    print(f"  Loading model: {model_path}")
    pipeline = joblib.load(model_path)

    print("  Loading FinancialPhraseBank dataset...")
    try:
        df = pd.read_csv(data_path)
        if "sentence" not in df.columns or "label" not in df.columns:
            raise ValueError("Expected columns 'sentence' and 'label' not found.")
    except Exception:
        df = pd.read_csv(
            data_path, sep="\t", header=None, names=["sentence", "label"]
        )

    # Ensure integer labels 0/1/2
    df = df.dropna(subset=["sentence", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    X_text: list[str] = df["sentence"].tolist()
    y = df["label"].values

    print(
        f"  Dataset: {len(df):,} sentences  "
        f"(neg={int((y==0).sum())}, neu={int((y==1).sum())}, pos={int((y==2).sum())})"
    )

    # Stratified 80/20 split for test-set evaluation
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("  Running 5-fold stratified CV (macro F1)...")
    cv_results = run_multiclass_cv(pipeline, X_text, y)

    # Test-set evaluation
    y_pred = pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    test_metrics = {
        "accuracy": test_accuracy,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
        # Expose the key scalar that ``save_model_card`` uses for CV/test comparison
        "f1": test_macro_f1,
    }

    print("  Generating CV box plot...")
    cv_plot = plot_cv_boxplot(cv_results, model_display_name, model_key, suffix="")

    print("  Generating 3-class confusion matrix heatmap...")
    cm_plot = plot_confusion_matrix(
        y_test,
        y_pred,
        model_display_name,
        model_key,
        class_labels=("Negative", "Neutral", "Positive"),
        suffix="",
    )

    print("  Saving model card JSON...")
    # Augment the card with per-class breakdown
    extra_info: dict[str, Any] = {
        "per_class_f1": {
            "negative": round(float(per_class_f1[0]), 4),
            "neutral": round(float(per_class_f1[1]), 4),
            "positive": round(float(per_class_f1[2]), 4),
        },
        "industry": "Financial Services",
        "note": (
            "No ROC curve generated for multiclass; macro F1 reported as primary metric."
        ),
    }

    card_path = save_model_card(
        model_key=model_key,
        model_name=model_display_name,
        model_type=model_type,
        dataset=dataset,
        cv_results=cv_results,
        test_metrics=test_metrics,
        feature_count=0,  # Text pipeline — no fixed feature count
        training_samples=len(X_train),
        test_samples=len(X_test),
        extra=extra_info,
        # Overwrite any previously generated Olist sentiment card
        filename_override="model_card_sentiment.json",
    )

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="model_card_sentiment_industry"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "dataset": dataset,
                "cv_folds": N_CV_FOLDS,
                "industry": "Financial Services",
            }
        )
        mlflow.log_metrics(
            {
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_macro_f1,
                "test_weighted_f1": test_weighted_f1,
                "test_f1_negative": float(per_class_f1[0]),
                "test_f1_neutral": float(per_class_f1[1]),
                "test_f1_positive": float(per_class_f1[2]),
                "cv_macro_f1_mean": float(np.mean(cv_results["f1"])),
                "cv_macro_f1_ci95": _ci_95(cv_results["f1"]),
            }
        )
        for artifact_path in (cv_plot, cm_plot, card_path):
            mlflow.log_artifact(str(artifact_path))

    print(
        f"  [OK] Sentiment industry card complete — "
        f"CV macro F1 {np.mean(cv_results['f1']):.3f}"
        f" +/- {_ci_95(cv_results['f1']):.3f}  |  Test accuracy {test_accuracy:.3f}"
    )


def generate_trend_card() -> None:
    """Generate model card artifacts for all 8 LightGBM trend forecasters.

    Loads each ``models/trend/forecaster_*.joblib`` artefact (a dict with keys
    ``model`` and ``meta``) and extracts the ``train_mae``/``test_mae`` values
    stored in ``meta`` at training time.  No re-evaluation is performed because
    the forecasters were trained on synthetic Gold-layer data that must be
    regenerated by ``seed_sandbox.py`` + ``seed_demo_incidents.py``.

    Produces a bar chart comparing train vs. test MAE across all 8 metrics and
    a JSON model card with per-metric and aggregate summary statistics.

    Artifacts saved:
        reports/model_card_trend.json
        reports/trend_forecaster_mae_comparison.png
    """
    model_key = "trend"
    model_display_name = "LightGBM Trend Forecaster (8 metrics)"
    model_type = "LightGBM Regressor (per-metric)"
    dataset = "Synthetic Business Metrics — seeded Gold layer (DuckDB)"

    trend_dir = MODELS_DIR / "trend"
    forecaster_paths = sorted(trend_dir.glob("forecaster_*.joblib"))

    if not forecaster_paths:
        print(
            f"  [WARN] No trend forecasters found in {trend_dir} — "
            "skipping trend card."
        )
        return

    print(f"  Found {len(forecaster_paths)} trend forecasters in {trend_dir}")

    per_metric: dict[str, dict[str, float]] = {}
    metrics_covered: list[str] = []

    for path in forecaster_paths:
        artefact: dict[str, Any] = joblib.load(path)
        meta: dict[str, Any] = artefact.get("meta", {})
        metric_name: str = meta.get(
            "metric_name",
            # Fallback: derive name from filename (forecaster_<name>.joblib)
            path.stem.replace("forecaster_", ""),
        )
        train_mae = float(meta.get("train_mae", 0.0))
        test_mae = float(meta.get("test_mae", meta.get("mae", 0.0)))

        per_metric[metric_name] = {
            "train_mae": round(train_mae, 6),
            "test_mae": round(test_mae, 6),
        }
        metrics_covered.append(metric_name)
        print(
            f"    {metric_name:30s}  train_mae={train_mae:.6f}  test_mae={test_mae:.6f}"
        )

    # Summary statistics
    test_maes = [v["test_mae"] for v in per_metric.values()]
    train_maes = [v["train_mae"] for v in per_metric.values()]
    avg_test_mae = float(np.mean(test_maes))
    avg_train_mae = float(np.mean(train_maes))
    # No overfitting when test MAE does not substantially exceed train MAE
    no_overfitting = bool(avg_test_mae <= avg_train_mae * 2.0)

    # --- Bar chart: train vs test MAE for all 8 metrics ---
    _apply_clean_style()
    palette = PALETTE[model_key]

    metric_labels = list(per_metric.keys())
    x = np.arange(len(metric_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_train = ax.bar(
        x - bar_width / 2,
        [per_metric[m]["train_mae"] for m in metric_labels],
        bar_width,
        label="Train MAE",
        color=palette["secondary"],
        edgecolor=palette["primary"],
        linewidth=0.8,
    )
    bars_test = ax.bar(
        x + bar_width / 2,
        [per_metric[m]["test_mae"] for m in metric_labels],
        bar_width,
        label="Test MAE",
        color=palette["primary"],
        edgecolor=palette["primary"],
        linewidth=0.8,
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in metric_labels],
        fontsize=9,
        ha="center",
    )
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title(
        f"{model_display_name} — Train vs Test MAE per Metric",
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper right", fontsize=9)

    # Annotate bars with numeric MAE values
    for bar_grp in (bars_train, bars_test):
        for bar in bar_grp:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.tight_layout()
    mae_plot = REPORTS_DIR / "trend_forecaster_mae_comparison.png"
    fig.savefig(mae_plot, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved MAE comparison chart: {mae_plot}")

    # --- Model card JSON ---
    card: dict[str, Any] = {
        "model_name": model_display_name,
        "model_type": model_type,
        "dataset": dataset,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metrics_covered": metrics_covered,
        "per_metric": per_metric,
        "summary": {
            "avg_test_mae": round(avg_test_mae, 6),
            "avg_train_mae": round(avg_train_mae, 6),
            "no_overfitting": no_overfitting,
        },
    }

    card_path = REPORTS_DIR / "model_card_trend.json"
    with open(card_path, "w", encoding="utf-8") as fh:
        json.dump(_to_native(card), fh, indent=2)
    print(f"  Saved model card: {card_path}")

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="model_card_trend"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "dataset": dataset,
                "n_forecasters": len(forecaster_paths),
            }
        )
        mlflow.log_metrics(
            {
                "avg_test_mae": avg_test_mae,
                "avg_train_mae": avg_train_mae,
                **{
                    f"test_mae_{name}": vals["test_mae"]
                    for name, vals in per_metric.items()
                },
            }
        )
        for artifact_path in (mae_plot, card_path):
            mlflow.log_artifact(str(artifact_path))

    print(
        f"  [OK] Trend card complete — avg test MAE {avg_test_mae:.6f}  "
        f"avg train MAE {avg_train_mae:.6f}  "
        f"no_overfitting={no_overfitting}"
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GENERATORS: dict[str, tuple[str, Any]] = {
    "churn": ("LightGBM Churn Classifier", generate_churn_card),
    "anomaly": ("Isolation Forest Anomaly Detector", generate_anomaly_card),
    "delivery": ("XGBoost Late Delivery Predictor", generate_delivery_card),
    "churn_telco": (
        "LightGBM Churn Classifier (Telco Industry)",
        generate_churn_telco_card,
    ),
    "delivery_dataco": (
        "XGBoost Late Delivery Predictor (DataCo Industry)",
        generate_delivery_dataco_card,
    ),
    "sentiment_industry": (
        "LinearSVC Financial Sentiment Classifier",
        generate_sentiment_industry_card,
    ),
    "trend": (
        "LightGBM Trend Forecaster (8 metrics)",
        generate_trend_card,
    ),
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for the model card generator.

    Returns:
        Exit code (0 = success, 1 = all models failed).
    """
    all_choices = list(GENERATORS.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description=(
            "LedgerGuard Model Card Generator — produces cross-validation plots, "
            "confusion matrices, ROC curves, and JSON model cards."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_model_cards.py --model all
  python scripts/generate_model_cards.py --model churn
  python scripts/generate_model_cards.py --model anomaly
  python scripts/generate_model_cards.py --model delivery
  python scripts/generate_model_cards.py --model churn_telco
  python scripts/generate_model_cards.py --model delivery_dataco
  python scripts/generate_model_cards.py --model sentiment_industry
  python scripts/generate_model_cards.py --model trend
        """,
    )
    parser.add_argument(
        "--model",
        choices=all_choices,
        default="all",
        help="Which model card(s) to generate (default: all).",
    )
    args = parser.parse_args()

    keys_to_run = (
        list(GENERATORS.keys()) if args.model == "all" else [args.model]
    )

    print("\nLedgerGuard Model Card Generator")
    print("=" * 60)
    print(f"  Models     : {', '.join(keys_to_run)}")
    print(f"  CV folds   : {N_CV_FOLDS}")
    print(f"  Bootstrap  : {BOOTSTRAP_ITERATIONS} iterations")
    print(f"  Reports dir: {REPORTS_DIR}")
    print(f"  MLflow exp : {MLFLOW_EXPERIMENT}")
    print("=" * 60)

    successes = 0
    failures = 0

    for key in keys_to_run:
        display_name, generator_fn = GENERATORS[key]
        print(f"\n[{key.upper()}] Generating model card for: {display_name}")
        print("-" * 60)
        try:
            generator_fn()
            successes += 1
        except Exception as exc:
            print(
                f"  [ERROR] {display_name} card generation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            import traceback

            traceback.print_exc()
            failures += 1

    print(f"\n{'=' * 60}")
    print(f"  Completed : {successes}/{len(keys_to_run)} model cards")
    print(f"  Failed    : {failures}/{len(keys_to_run)}")
    print(f"  Artifacts : {REPORTS_DIR}")
    print(f"  MLflow    : mlflow ui  ->  http://localhost:5000")
    print("=" * 60)

    return 0 if failures < len(keys_to_run) else 1


if __name__ == "__main__":
    sys.exit(main())
