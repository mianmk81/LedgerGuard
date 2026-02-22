"""
Late Delivery Risk Predictor — supports DataCo XGBoost and Olist stacked ensemble.

Model priority:
1. Stacked ensemble (models/delivery/stacked_ensemble_late_delivery.joblib)
2. Two-stage model (models/delivery/two_stage_late_delivery.joblib)
3. DataCo XGBoost (models/delivery_industry/xgboost_dataco_delivery.joblib)
"""

import structlog
from pathlib import Path
from typing import Any, Optional

from api.config import get_settings

logger = structlog.get_logger()

_DELIVERY_MODEL_CACHE: Optional[dict] = None
_STACKED_MODEL_CACHE: Optional[dict] = None
_TWO_STAGE_MODEL_CACHE: Optional[dict] = None

# Default feature values when client omits them
DELIVERY_DEFAULT_FEATURES = {
    "order_day_of_week": 2,
    "order_month": 6,
    "order_hour": 14,
    "order_day_of_month": 15,
    "is_weekend": 0,
    "scheduled_shipping_days": 4,
    "shipping_mode_encoded": 0,
    "sales_per_customer": 100,
    "item_discount": 0,
    "discount_rate": 0,
    "profit_ratio": 0.1,
    "item_quantity": 2,
    "sales": 50,
    "profit_per_order": 5,
    "product_price": 25,
    "customer_segment_encoded": 0,
    "market_encoded": 0,
    "region_encoded": 0,
    "category_encoded": 0,
    "order_status_encoded": 0,
    "value_per_item": 25,
    "discount_amount": 0,
    "high_value_order": 0,
    "rush_shipping": 0,
    "high_quantity": 0,
}


def _load_delivery_model() -> Optional[dict]:
    """Lazy-load XGBoost delivery model (train_industry_delivery.py)."""
    global _DELIVERY_MODEL_CACHE
    if _DELIVERY_MODEL_CACHE is not None:
        return _DELIVERY_MODEL_CACHE
    try:
        import joblib

        settings = get_settings()
        path = Path(settings.models_dir) / "delivery_industry" / "xgboost_dataco_delivery.joblib"
        if path.exists():
            _DELIVERY_MODEL_CACHE = joblib.load(path)
            logger.info("delivery_model_loaded", path=str(path))
            return _DELIVERY_MODEL_CACHE
    except Exception as e:
        logger.warning("delivery_model_load_failed", error=str(e))
    return None


def _load_stacked_model() -> Optional[dict]:
    """Lazy-load stacked ensemble delivery model."""
    global _STACKED_MODEL_CACHE
    if _STACKED_MODEL_CACHE is not None:
        return _STACKED_MODEL_CACHE
    try:
        import joblib

        settings = get_settings()
        path = Path(settings.models_dir) / "delivery" / "stacked_ensemble_late_delivery.joblib"
        if path.exists():
            _STACKED_MODEL_CACHE = joblib.load(path)
            logger.info("stacked_delivery_model_loaded", path=str(path))
            return _STACKED_MODEL_CACHE
    except Exception as e:
        logger.warning("stacked_delivery_model_load_failed", error=str(e))
    return None


def _load_two_stage_model() -> Optional[dict]:
    """Lazy-load two-stage delivery model."""
    global _TWO_STAGE_MODEL_CACHE
    if _TWO_STAGE_MODEL_CACHE is not None:
        return _TWO_STAGE_MODEL_CACHE
    try:
        import joblib

        settings = get_settings()
        path = Path(settings.models_dir) / "delivery" / "two_stage_late_delivery.joblib"
        if path.exists():
            _TWO_STAGE_MODEL_CACHE = joblib.load(path)
            logger.info("two_stage_delivery_model_loaded", path=str(path))
            return _TWO_STAGE_MODEL_CACHE
    except Exception as e:
        logger.warning("two_stage_delivery_model_load_failed", error=str(e))
    return None


def _predict_with_stacked(order_features: dict[str, Any], artifact: dict) -> Optional[dict]:
    """Predict using stacked ensemble. Returns None if features don't match."""
    try:
        import numpy as np

        base_models = artifact["base_models"]
        meta_learner = artifact["meta_learner"]
        threshold = artifact["threshold"]

        # Stacked model needs base model predictions as meta-features
        # Build feature vector for each base model
        import pandas as pd

        # Get features from first base model
        first_model = list(base_models.values())[0]
        if hasattr(first_model, "feature_names_in_"):
            features = list(first_model.feature_names_in_)
        else:
            return None

        row = {f: order_features.get(f, 0) for f in features}
        X = pd.DataFrame([row]).reindex(columns=features, fill_value=0)

        # Get base model predictions
        base_preds = []
        for m in base_models.values():
            base_preds.append(float(m.predict_proba(X)[0, 1]))
        meta_X = np.array([base_preds])

        proba = float(meta_learner.predict_proba(meta_X)[0, 1])
        is_late = proba >= threshold

        return {
            "is_late_predicted": bool(is_late),
            "probability": round(proba, 4),
            "threshold": threshold,
            "model_used": True,
            "model_name": artifact.get("model_name", "stacked_ensemble_late_delivery"),
            "model_version": artifact.get("model_version", "1.0"),
        }
    except Exception as e:
        logger.warning("stacked_predict_failed", error=str(e))
        return None


def predict_delivery_risk(order_features: dict[str, Any]) -> dict[str, Any]:
    """
    Predict late delivery risk for a single order.

    Tries models in priority order: stacked ensemble > two-stage > DataCo XGBoost.

    Args:
        order_features: Dict with feature keys. Missing keys filled with defaults.

    Returns:
        {"is_late_predicted": bool, "probability": float, "model_used": bool}
    """
    # Try stacked ensemble first (best expected performance)
    stacked = _load_stacked_model()
    if stacked is not None:
        result = _predict_with_stacked(order_features, stacked)
        if result is not None:
            return result

    # Fall back to DataCo XGBoost
    model_artifact = _load_delivery_model()
    if model_artifact is None:
        return {
            "is_late_predicted": False,
            "probability": 0.0,
            "model_used": False,
            "message": "Delivery model not found. Run: python scripts/train_late_delivery.py --model all",
        }

    model = model_artifact["model"]
    threshold = model_artifact["threshold"]
    features = model_artifact["features"]

    # Build feature vector
    row = {}
    for f in features:
        row[f] = order_features.get(f, DELIVERY_DEFAULT_FEATURES.get(f, 0))
    try:
        import pandas as pd

        X = pd.DataFrame([row])
        X = X.reindex(columns=features, fill_value=0)
        proba = float(model.predict_proba(X)[0, 1])
        is_late = proba >= threshold
        return {
            "is_late_predicted": bool(is_late),
            "probability": round(proba, 4),
            "threshold": threshold,
            "model_used": True,
            "model_name": model_artifact.get("model_name", "xgboost_dataco_delivery"),
            "model_version": model_artifact.get("model_version", "1.0"),
        }
    except Exception as e:
        logger.warning("delivery_predict_failed", error=str(e))
        return {
            "is_late_predicted": False,
            "probability": 0.0,
            "model_used": False,
            "error": str(e),
        }


def predict_delivery_batch(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict late delivery risk for multiple orders."""
    return [predict_delivery_risk(o) for o in orders]
