"""
System health and diagnostics router.

Wired to:
- StorageBackend for database diagnostics
- Settings for configuration
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse

from api.config import get_settings
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Track startup time for uptime calculation
_startup_time = time.time()

# Resolve project root once at import time: two levels up from this file
# (api/routers/system.py -> api/routers -> api -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@router.get("/health")
async def system_health():
    """
    Get system health status.
    Checks database connectivity and reports actual service health.
    """
    settings = get_settings()
    uptime = time.time() - _startup_time

    # Check database health
    db_status = "healthy"
    try:
        from api.storage import get_storage

        storage = get_storage()
        # Simple read to verify connectivity
        storage.read_monitors(enabled=True)
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "success": True,
        "data": {
            "status": "healthy" if db_status == "healthy" else "degraded",
            "version": "0.1.0",
            "uptime_seconds": round(uptime, 1),
            "database": db_status,
            "environment": settings.intuit_env,
        },
    }


@router.get("/diagnostics")
async def system_diagnostics():
    """
    Get detailed system diagnostics.
    Reports database size, table counts, and engine status.
    """
    settings = get_settings()

    logger.info("diagnostics_request")

    diagnostics = {
        "environment": settings.intuit_env,
        "database_type": settings.db_type,
        "database_path": settings.db_path,
        "database_size_mb": 0.0,
        "tables": {},
    }

    # Check DB file size
    try:
        if os.path.exists(settings.db_path):
            size_bytes = os.path.getsize(settings.db_path)
            diagnostics["database_size_mb"] = round(size_bytes / (1024 * 1024), 2)
    except Exception:
        pass

    # Get table row counts
    try:
        from api.storage import get_storage

        storage = get_storage()

        diagnostics["tables"]["incidents"] = len(storage.read_incidents())
        diagnostics["tables"]["monitors"] = len(storage.read_monitors())
        diagnostics["tables"]["cascades"] = len(storage.read_cascades())
        diagnostics["tables"]["alerts"] = len(storage.read_monitor_alerts())
    except Exception as e:
        diagnostics["tables"]["error"] = str(e)

    return {"success": True, "data": diagnostics}


@router.get("/config")
async def get_system_config():
    """
    Get system configuration (non-sensitive values only).
    """
    settings = get_settings()

    return {
        "success": True,
        "data": {
            "environment": settings.intuit_env,
            "log_level": settings.log_level,
            "db_type": settings.db_type,
            "anomaly_detection_sensitivity": settings.anomaly_detection_sensitivity,
            "min_confidence_threshold": settings.min_confidence_threshold,
            "blast_radius_max_depth": settings.blast_radius_max_depth,
            "monitor_evaluation_interval_seconds": settings.monitor_evaluation_interval_seconds,
            "feature_flags": {
                "ml_training": settings.enable_ml_training,
                "auto_remediation": settings.enable_auto_remediation,
                "realtime_analysis": settings.enable_realtime_analysis,
                "supplemental_data": settings.enable_supplemental_data,
            },
        },
    }


def _get_file_size_mb(path: Path) -> float | None:
    """
    Return the size of a file in megabytes, rounded to 3 decimal places.

    Args:
        path: Absolute path to the file.

    Returns:
        Size in MB if the path is an existing regular file, otherwise None.
    """
    try:
        if path.is_file():
            return round(path.stat().st_size / (1024 * 1024), 3)
    except OSError:
        pass
    return None


def _probe_file_model(
    name: str,
    relative_path: str,
    task: str,
    dataset: str,
) -> dict[str, Any]:
    """
    Build a status record for a single file-based model artifact.

    Status values:
    - ``"available"``  — file exists and is readable
    - ``"missing"``    — file does not exist
    - ``"error"``      — path exists but could not be stat'd

    Args:
        name: Human-readable model name shown in the API response.
        relative_path: Path relative to the project root (forward-slash notation).
        task: High-level ML task this model serves (e.g. ``"churn"``).
        dataset: Dataset the model was trained on (e.g. ``"olist"``).

    Returns:
        Dictionary with keys: name, path, status, size_mb, task, dataset.
    """
    abs_path = _PROJECT_ROOT / relative_path
    size_mb = _get_file_size_mb(abs_path)

    if size_mb is not None:
        status = "available"
    elif abs_path.exists():
        # Exists (possibly a broken symlink or permission error)
        status = "error"
    else:
        status = "missing"

    return {
        "name": name,
        "path": relative_path,
        "status": status,
        "size_mb": size_mb,
        "task": task,
        "dataset": dataset,
    }


def _probe_directory_model(
    name: str,
    relative_dir: str,
    task: str,
    dataset: str,
) -> dict[str, Any]:
    """
    Build a status record for a directory-based model artifact.

    A directory is considered ``"available"`` when it exists and contains at
    least one file.  An empty directory or a missing path is reported as
    ``"missing"``.

    Args:
        name: Human-readable model name shown in the API response.
        relative_dir: Directory path relative to the project root.
        task: High-level ML task this model serves.
        dataset: Dataset the model was trained on.

    Returns:
        Dictionary with keys: name, path, status, size_mb, task, dataset.
    """
    abs_dir = _PROJECT_ROOT / relative_dir

    try:
        if abs_dir.is_dir():
            files = list(abs_dir.iterdir())
            if files:
                total_bytes = sum(
                    f.stat().st_size for f in files if f.is_file()
                )
                size_mb: float | None = round(total_bytes / (1024 * 1024), 3)
                status = "available"
            else:
                # Directory exists but is empty — no usable artifact
                size_mb = None
                status = "missing"
        else:
            size_mb = None
            status = "missing"
    except OSError:
        size_mb = None
        status = "error"

    return {
        "name": name,
        "path": relative_dir,
        "status": status,
        "size_mb": size_mb,
        "task": task,
        "dataset": dataset,
    }


def _probe_runtime_model(
    name: str,
    task: str,
    dataset: str,
    description: str,
) -> dict[str, Any]:
    """
    Build a status record for a runtime-trained model (no persisted artifact).

    The ``MLDetector`` trains an Isolation Forest (and optionally a full
    4-model ensemble) in memory each time the detection engine runs.  There is
    no file to check; the status is always reported as ``"loaded"`` to indicate
    the class is available for instantiation.

    Args:
        name: Human-readable model name.
        task: High-level ML task (e.g. ``"anomaly"``).
        dataset: Dataset / data source used at runtime.
        description: Short human-readable note about the runtime behaviour.

    Returns:
        Dictionary with keys: name, path, status, size_mb, task, dataset,
        and an extra ``note`` field.
    """
    # Verify the MLDetector class itself can be imported so we surface import
    # errors immediately rather than silently claiming the model is ready.
    try:
        from api.engine.detection.ml_detector import MLDetector  # noqa: F401

        status = "loaded"
        note = description
    except Exception as exc:  # pragma: no cover
        status = "error"
        note = f"Import error: {exc}"

    return {
        "name": name,
        "path": None,
        "status": status,
        "size_mb": None,
        "task": task,
        "dataset": dataset,
        "note": note,
    }


@router.get("/models")
async def get_model_status():
    """
    Report availability and file sizes for all ML model artifacts.

    Checks the filesystem for each expected model file or directory and
    returns a per-model status alongside an aggregate summary.

    Status values per model:
    - ``"loaded"``    — runtime model; class importable (no file artifact)
    - ``"available"`` — artifact file/directory exists on disk
    - ``"missing"``   — artifact file/directory not found
    - ``"error"``     — artifact path exists but could not be read

    Returns:
        Standard ``{success, data}`` envelope where ``data`` contains
        ``models`` (list of model status dicts) and ``summary`` (aggregate
        counts).
    """
    logger.info("model_status_request")

    models: list[dict[str, Any]] = [
        # --- Sentiment (industry-level: LinearSVC preferred, TF-IDF+LR fallback) ---
        _probe_file_model(
            name="Sentiment — LinearSVC (FinancialPhraseBank, Industry)",
            relative_path="models/sentiment_industry/linear_svc_financial_sentiment.joblib",
            task="sentiment",
            dataset="financial_phrasebank",
        ),
        _probe_file_model(
            name="Sentiment — TF-IDF + LR (FinancialPhraseBank, Fallback)",
            relative_path="models/sentiment_industry/tfidf_lr_financial_sentiment.joblib",
            task="sentiment",
            dataset="financial_phrasebank",
        ),
        # --- Late-delivery: industry (DataCo) ---
        _probe_file_model(
            name="Late Delivery — XGBoost (DataCo Industry)",
            relative_path="models/delivery_industry/xgboost_dataco_delivery.joblib",
            task="delivery",
            dataset="dataco",
        ),
        _probe_file_model(
            name="Late Delivery — Random Forest (DataCo Industry)",
            relative_path="models/delivery_industry/random_forest_dataco_delivery.joblib",
            task="delivery",
            dataset="dataco",
        ),
        _probe_file_model(
            name="Late Delivery — Logistic Regression (DataCo Industry)",
            relative_path="models/delivery_industry/logistic_regression_dataco_delivery.joblib",
            task="delivery",
            dataset="dataco",
        ),
        # --- Churn: industry (Telco) ---
        _probe_file_model(
            name="Churn — LightGBM (Telco Industry)",
            relative_path="models/churn_industry/lightgbm_telco_churn.pkl",
            task="churn",
            dataset="telco",
        ),
        # --- Churn: Olist ---
        _probe_file_model(
            name="Churn — LightGBM (Olist)",
            relative_path="models/churn/lightgbm_churn_model.pkl",
            task="churn",
            dataset="olist",
        ),
        _probe_file_model(
            name="Churn — Random Forest (Olist)",
            relative_path="models/churn/random_forest_churn_model.pkl",
            task="churn",
            dataset="olist",
        ),
        # --- Anomaly: industry directory ---
        _probe_directory_model(
            name="Anomaly Detection — Industry Ensemble (directory)",
            relative_dir="models/anomaly_industry",
            task="anomaly",
            dataset="industry",
        ),
        # --- Anomaly: runtime isolation forest ---
        _probe_runtime_model(
            name="Anomaly Detection — Runtime Isolation Forest (MLDetector)",
            task="anomaly",
            dataset="runtime_business_metrics",
            description=(
                "4-model strict ensemble (IF + OCSVM + LOF + Autoencoder) trained "
                "in memory on historical business metrics at detection time. "
                "No persisted file artifact."
            ),
        ),
        # --- Late-delivery: Olist ---
        _probe_file_model(
            name="Late Delivery — XGBoost (Olist)",
            relative_path="models/delivery/xgboost_late_delivery.joblib",
            task="delivery",
            dataset="olist",
        ),
        # --- Causal graph: Granger-learned edge strengths ---
        _probe_file_model(
            name="Causal Graph — Granger Causality (Gold Metrics)",
            relative_path="models/causal_graph/granger_edges.json",
            task="rca",
            dataset="gold_daily_metrics",
        ),
    ]

    # Build summary counts
    available_statuses = {"loaded", "available"}
    total = len(models)
    available_count = sum(1 for m in models if m["status"] in available_statuses)
    missing_count = sum(1 for m in models if m["status"] == "missing")

    logger.info(
        "model_status_complete",
        total=total,
        available=available_count,
        missing=missing_count,
    )

    return {
        "success": True,
        "data": {
            "models": models,
            "summary": {
                "total_models": total,
                "available": available_count,
                "missing": missing_count,
            },
        },
    }


@router.get("/experiments")
async def get_mlflow_experiments():
    """
    Return MLflow experiment summaries for the top models.

    Reads from the local MLflow tracking store and returns per-experiment
    run counts, best metrics, and parameters of the best run.
    """
    logger.info("experiments_request")

    experiments_config = [
        {
            "name": "ledgerguard-churn-prediction",
            "display": "Churn Prediction",
            "primary_metric": "f1",
            "higher_is_better": True,
        },
        {
            "name": "ledgerguard-anomaly-detection",
            "display": "Anomaly Detection",
            "primary_metric": "test_f1",
            "higher_is_better": True,
        },
        {
            "name": "ledgerguard-late-delivery",
            "display": "Late Delivery Prediction",
            "primary_metric": "test_f1",
            "higher_is_better": True,
        },
    ]

    results = []
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()

        for exp_cfg in experiments_config:
            exp = client.get_experiment_by_name(exp_cfg["name"])
            if not exp:
                results.append(
                    {
                        "experiment": exp_cfg["display"],
                        "mlflow_name": exp_cfg["name"],
                        "status": "not_found",
                        "run_count": 0,
                        "best_run": None,
                    }
                )
                continue

            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=[
                    f"metrics.{exp_cfg['primary_metric']} "
                    f"{'DESC' if exp_cfg['higher_is_better'] else 'ASC'}"
                ],
                max_results=20,
            )

            if not runs:
                results.append(
                    {
                        "experiment": exp_cfg["display"],
                        "mlflow_name": exp_cfg["name"],
                        "status": "empty",
                        "run_count": 0,
                        "best_run": None,
                    }
                )
                continue

            best = runs[0]
            results.append(
                {
                    "experiment": exp_cfg["display"],
                    "mlflow_name": exp_cfg["name"],
                    "status": "active",
                    "run_count": len(runs),
                    "best_run": {
                        "run_id": best.info.run_id,
                        "metrics": dict(best.data.metrics),
                        "params": dict(best.data.params),
                        "start_time": best.info.start_time,
                        "model_type": best.data.params.get("model_type", "unknown"),
                    },
                    "all_runs": [
                        {
                            "run_id": r.info.run_id,
                            "model_type": r.data.params.get("model_type", "unknown"),
                            "metrics": {
                                k: v
                                for k, v in r.data.metrics.items()
                                if k.startswith("test_") or k in ("f1", "auc_roc", "accuracy")
                            },
                        }
                        for r in runs
                    ],
                }
            )

    except ImportError:
        logger.warning("mlflow_not_installed")
        return {
            "success": False,
            "error": "MLflow is not installed",
            "data": None,
        }
    except Exception as exc:
        logger.error("experiments_error", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "data": None,
        }

    logger.info("experiments_complete", count=len(results))
    return {"success": True, "data": {"experiments": results}}


@router.get("/reports/{filename}")
async def get_report_file(filename: str):
    """
    Serve a generated report image (PNG/JSON) from the reports/ directory.
    """
    # Sanitize: only allow alphanumeric, underscore, hyphen, dot
    import re

    if not re.match(r"^[\w\-]+\.(png|json|jpg|svg)$", filename):
        return {"success": False, "error": "Invalid filename"}

    filepath = _PROJECT_ROOT / "reports" / filename
    if not filepath.exists():
        return {"success": False, "error": "File not found"}

    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".json": "application/json",
    }
    media_type = media_types.get(filepath.suffix, "application/octet-stream")
    return FileResponse(str(filepath), media_type=media_type)


@router.get("/model-cards")
async def get_model_cards():
    """
    Return model card JSON data for all models that have generated cards.
    """
    logger.info("model_cards_request")
    reports_dir = _PROJECT_ROOT / "reports"
    cards = {}

    if reports_dir.exists():
        for card_file in reports_dir.glob("model_card_*.json"):
            try:
                with open(card_file) as f:
                    card_data = json.load(f)
                model_key = card_file.stem.replace("model_card_", "")
                cards[model_key] = card_data
            except Exception as exc:
                logger.warning("model_card_read_error", file=str(card_file), error=str(exc))

    # Also list available report images
    images = {}
    if reports_dir.exists():
        for img_file in reports_dir.glob("*.png"):
            category = img_file.stem.rsplit("_", 1)[-1] if "_" in img_file.stem else "other"
            if category not in images:
                images[category] = []
            images[category].append(img_file.name)

    return {
        "success": True,
        "data": {
            "cards": cards,
            "report_images": {
                name: sorted(files) for name, files in sorted(images.items())
            },
        },
    }
