"""
ML Prediction Router — Order delivery risk and customer churn risk.

Uses trained models from:
- models/delivery_industry/ (train_industry_delivery.py)
- models/churn/ (train_churn_model.py)
"""

from typing import Any

from fastapi import APIRouter, Depends

from api.auth.dependencies import get_current_realm_id
from api.engine.prediction.delivery_predictor import (
    DELIVERY_DEFAULT_FEATURES,
    predict_delivery_batch,
    predict_delivery_risk,
)
from api.engine.prediction.churn_predictor import (
    CHURN_DEFAULT_FEATURES,
    predict_churn_batch,
    predict_churn_risk,
)
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/delivery-risk")
async def predict_order_delivery_risk(
    order: dict[str, Any],
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Predict late delivery risk for a single order.

    Send order features (DataCo schema). Example:
    {
      "scheduled_shipping_days": 4,
      "shipping_mode_encoded": 0,
      "sales": 100,
      "order_day_of_week": 2,
      ...
    }
    Missing features use sensible defaults.
    """
    logger.info("prediction_delivery_single", realm_id=realm_id)
    return {"success": True, "data": predict_delivery_risk(order)}


@router.post("/delivery-risk/batch")
async def predict_order_delivery_risk_batch(
    orders: list[dict[str, Any]],
    realm_id: str = Depends(get_current_realm_id),
):
    """Predict late delivery risk for multiple orders."""
    logger.info("prediction_delivery_batch", realm_id=realm_id, count=len(orders))
    return {"success": True, "data": predict_delivery_batch(orders)}


@router.get("/delivery-risk/schema")
async def get_delivery_schema(realm_id: str = Depends(get_current_realm_id)):
    """Return expected delivery prediction features and defaults."""
    return {
        "success": True,
        "data": {
            "features": list(DELIVERY_DEFAULT_FEATURES.keys()),
            "defaults": DELIVERY_DEFAULT_FEATURES,
        },
    }


@router.post("/churn-risk")
async def predict_customer_churn_risk(
    customer: dict[str, Any],
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Predict churn probability for a single customer.

    Send customer features (Olist RFM schema). Example:
    {
      "recency_days": 45,
      "frequency": 5,
      "monetary": 800,
      "avg_review_score": 4.2,
      ...
    }
    """
    logger.info("prediction_churn_single", realm_id=realm_id)
    return {"success": True, "data": predict_churn_risk(customer)}


@router.post("/churn-risk/batch")
async def predict_customer_churn_risk_batch(
    customers: list[dict[str, Any]],
    realm_id: str = Depends(get_current_realm_id),
):
    """Predict churn risk for multiple customers."""
    logger.info("prediction_churn_batch", realm_id=realm_id, count=len(customers))
    return {"success": True, "data": predict_churn_batch(customers)}


@router.get("/churn-risk/schema")
async def get_churn_schema(realm_id: str = Depends(get_current_realm_id)):
    """Return expected churn prediction features and defaults."""
    return {
        "success": True,
        "data": {
            "features": list(CHURN_DEFAULT_FEATURES.keys()),
            "defaults": CHURN_DEFAULT_FEATURES,
        },
    }
