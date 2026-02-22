"""
Incident Detection Engine for Business Reliability Engineering.

This module provides multi-layered anomaly detection combining statistical methods,
machine learning, and changepoint detection to identify operational incidents across
financial, operational, and customer health domains.

Detection Architecture:
    Layer 1: Statistical Detection (MAD-based Z-score) - Always runs
    Layer 2: ML Detection (Isolation Forest) - Runs if enabled
    Layer 3: Changepoint Detection (Ruptures/PELT) - Runs if enabled

    Results are fused using ensemble logic to produce high-confidence incidents
    with typed classifications across 8 incident types.

Components:
    StatisticalDetector: MAD-based z-score anomaly detection
    MLDetector: 4-model strict ensemble (IF+OCSVM+LOF+AE, F1 0.20)
    ChangepointDetector: PELT algorithm regime change detection
    EnsembleDetector: Detection fusion and incident creation
"""

from api.engine.detection.changepoint import ChangepointDetector
from api.engine.detection.ensemble import EnsembleDetector
from api.engine.detection.ml_detector import MLDetector
from api.engine.detection.statistical import StatisticalDetector

__all__ = [
    "StatisticalDetector",
    "MLDetector",
    "ChangepointDetector",
    "EnsembleDetector",
]
