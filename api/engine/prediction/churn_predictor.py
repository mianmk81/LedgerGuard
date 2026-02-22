"""
Churn Risk Predictor — Industry (Telco) and Olist-trained models.

Prefers industry-level Telco LightGBM (AUC 0.85) when available; falls back to
Olist models. Telco model uses Olist→Telco feature adapter (RFM mapped to
tenure/charges/contract proxy).
"""

import structlog
from pathlib import Path
from typing import Any, Optional

from api.config import get_settings

logger = structlog.get_logger()

_CHURN_MODEL_CACHE: Optional[dict] = None

# Churn feature defaults (Olist schema)
CHURN_DEFAULT_FEATURES = {
    "recency_days": 30,
    "frequency": 3,
    "monetary": 500,
    "log_monetary": 6.2,
    "recency_squared": 900,
    "recency_frequency": 90,
    "avg_order_value": 167,
    "lifespan_days": 60,
    "purchase_rate": 0.05,
    "avg_review_score": 4.0,
    "avg_delivery_days": 12,
    "late_order_count": 0,
    "late_order_rate": 0,
    "avg_installments": 2,
    "product_diversity": 2,
    "total_items": 5,
    "complaint_count": 0,
}

# Telco feature schema (industry model) — map Olist RFM to approximate Telco equivalents
TELCO_FEATURE_DEFAULTS = {
    "gender_encoded": 0,
    "SeniorCitizen_encoded": 0,
    "Partner_encoded": 0,
    "Dependents_encoded": 0,
    "tenure": 2,
    "MonthlyCharges": 70,
    "TotalCharges": 140,
    "contract_encoded": 0,
    "PaperlessBilling_encoded": 0,
    "pay_electronic": 0,
    "pay_mailed": 0,
    "pay_bank_transfer": 0,
    "pay_credit_card": 1,
    "PhoneService_encoded": 1,
    "MultipleLines_encoded": 0,
    "internet_dsl": 0,
    "internet_fiber": 0,
    "internet_none": 1,
    "OnlineSecurity_encoded": 0,
    "OnlineBackup_encoded": 0,
    "DeviceProtection_encoded": 0,
    "TechSupport_encoded": 0,
    "StreamingTV_encoded": 0,
    "StreamingMovies_encoded": 0,
    "num_premium_services": 0,
    "charges_per_tenure": 35,
    "charges_ratio": 1.0,
    "tenure_bucket": 0,
    "high_value": 0,
    "at_risk_combo": 0,
}


def _olist_to_telco_features(olist: dict[str, Any]) -> dict[str, Any]:
    """Map Olist RFM features to approximate Telco schema for industry model."""
    lifespan = max(0, olist.get("lifespan_days", 60))
    freq = max(1, olist.get("frequency", 3))
    monetary = float(olist.get("monetary", 500))
    avg_order = float(olist.get("avg_order_value", monetary / freq))
    tenure_months = max(0, lifespan // 30)
    return {
        **TELCO_FEATURE_DEFAULTS,
        "tenure": tenure_months,
        "MonthlyCharges": avg_order,
        "TotalCharges": monetary,
        "charges_per_tenure": avg_order / (tenure_months + 1),
        "charges_ratio": monetary / (tenure_months * avg_order + 1),
        "tenure_bucket": min(3, tenure_months // 12),
        "high_value": 1 if avg_order > 100 else 0,
    }


def _load_churn_model() -> Optional[dict]:
    """Lazy-load churn model. Prefer Telco (industry) > Olist LightGBM > RF > LogReg."""
    global _CHURN_MODEL_CACHE
    if _CHURN_MODEL_CACHE is not None:
        return _CHURN_MODEL_CACHE
    try:
        import joblib

        settings = get_settings()
        base = Path(settings.models_dir)

        # 1. Try industry Telco model (AUC 0.85)
        telco_path = base / "churn_industry" / "lightgbm_telco_churn.pkl"
        if telco_path.exists():
            art = joblib.load(telco_path)
            art["feature_names"] = art.get("features", art.get("feature_names", []))
            art["schema"] = "telco"
            art["model_name"] = "churn_telco_industry"
            _CHURN_MODEL_CACHE = art
            logger.info("churn_model_loaded", path=str(telco_path), schema="telco")
            return _CHURN_MODEL_CACHE

        # 2. Fallback to Olist models
        models_dir = base / "churn"
        for name in ["lightgbm_churn_model", "random_forest_churn_model", "logistic_regression_churn_model"]:
            path = models_dir / f"{name}.pkl"
            if path.exists():
                art = joblib.load(path)
                art["schema"] = "olist"
                art["feature_names"] = art.get("feature_names", art.get("features", []))
                _CHURN_MODEL_CACHE = art
                logger.info("churn_model_loaded", path=str(path), schema="olist")
                return _CHURN_MODEL_CACHE
    except Exception as e:
        logger.warning("churn_model_load_failed", error=str(e))
    return None


def predict_churn_risk(customer_features: dict[str, Any]) -> dict[str, Any]:
    """
    Predict churn probability for a single customer.

    Args:
        customer_features: Dict with Olist churn schema keys. Missing keys filled with defaults.

    Returns:
        {"churn_predicted": bool, "probability": float, "model_used": bool}
    """
    model_artifact = _load_churn_model()
    if model_artifact is None:
        return {
            "churn_predicted": False,
            "probability": 0.0,
            "model_used": False,
            "message": "Churn model not found. Run: python scripts/train_industry_churn.py --model lgbm (Telco) or python scripts/train_churn_model.py --model lgbm (Olist)",
        }

    model = model_artifact["model"]
    threshold = model_artifact.get("threshold", 0.5)
    feature_names = model_artifact.get("feature_names") or model_artifact.get("features", [])
    schema = model_artifact.get("schema", "olist")

    if schema == "telco":
        row = _olist_to_telco_features(customer_features)
    else:
        row = {f: customer_features.get(f, CHURN_DEFAULT_FEATURES.get(f, 0)) for f in feature_names}

    # Ensure all feature_names present (fill missing with 0)
    for f in feature_names:
        if f not in row:
            row[f] = 0

    try:
        import pandas as pd

        X = pd.DataFrame([row])
        X = X.reindex(columns=feature_names, fill_value=0)
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
        else:
            proba = float(model.predict(X)[0])
        is_churn = proba >= threshold
        return {
            "churn_predicted": bool(is_churn),
            "probability": round(proba, 4),
            "threshold": threshold,
            "model_used": True,
            "model_name": model_artifact.get("model_name", "churn_lgbm"),
            "model_version": model_artifact.get("model_version", "1.0"),
        }
    except Exception as e:
        logger.warning("churn_predict_failed", error=str(e))
        return {
            "churn_predicted": False,
            "probability": 0.0,
            "model_used": False,
            "error": str(e),
        }


def predict_churn_batch(customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict churn risk for multiple customers."""
    return [predict_churn_risk(c) for c in customers]
