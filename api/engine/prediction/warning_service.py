"""
Warning Service — Orchestrates trend detection, forward chain, and prevention.

Agent: backend-developer
Computes early warnings on-demand from Gold metrics. No persistence.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from api.models.early_warning import EarlyWarning
from api.engine.prediction.trend_detector import TrendDetector
from api.engine.prediction.forward_chain import ForwardChainBuilder
from api.storage.base import StorageBackend

logger = structlog.get_logger()


def _default_models_dir() -> Optional[Path]:
    """Resolve models/trend from db path for trained forecasters."""
    try:
        from api.config import get_settings

        db_path = Path(get_settings().db_path).resolve()
        return db_path.parent.parent / "models" / "trend"
    except Exception:
        return None


class WarningService:
    """
    Generates early warnings from trend detection + forward chain analysis.
    Uses trained LightGBM forecasters when available (run train_trend_forecaster.py).
    """

    def __init__(
        self,
        storage: StorageBackend,
        lookback_days: int = 14,
        projection_days: int = 5,
        models_dir: Optional[Path | str] = None,
    ):
        self.storage = storage
        self.trend_detector = TrendDetector(
            storage=storage,
            lookback_days=lookback_days,
            projection_days=projection_days,
            models_dir=models_dir or _default_models_dir(),
        )
        self.chain_builder = ForwardChainBuilder()
        self.logger = structlog.get_logger()

    def get_active_warnings(self) -> list[EarlyWarning]:
        """
        Compute and return active early warnings.

        Returns:
            List of EarlyWarning instances
        """
        degrading = self.trend_detector.detect_degrading_metrics()
        warnings = []

        for d in degrading:
            chains = self.chain_builder.build_chain(d["metric"])

            # Use first chain if any, else generic prevention
            if chains:
                chain_info = chains[0]
                path_with_incident = chain_info["path"] + [chain_info["incident_type"]]
                prevention = chain_info["prevention_steps"]
                incident_type = chain_info["incident_type"]
            else:
                path_with_incident = [d["metric"]]
                prevention = f"Monitor {d['metric'].replace('_', ' ')} to prevent incident."
                incident_type = None

            # Severity from days to threshold
            days = d.get("days_to_threshold")
            if days is not None:
                if days <= 2:
                    severity = "critical"
                elif days <= 5:
                    severity = "high"
                elif days <= 10:
                    severity = "medium"
                else:
                    severity = "low"
            else:
                severity = "medium"

            warning = EarlyWarning(
                metric=d["metric"],
                current_value=d["current_value"],
                baseline=d.get("baseline"),
                slope=d["slope"],
                projected_value=d["projected_value"],
                projected_ci_lower=d.get("projected_ci_lower"),
                projected_ci_upper=d.get("projected_ci_upper"),
                p_value=d.get("p_value"),
                projection_days=self.trend_detector.projection_days,
                threshold=d["threshold"],
                threshold_below=d["threshold_below"],
                forward_chain=path_with_incident,
                incident_type=incident_type,
                prevention_steps=prevention,
                severity=severity,
                days_to_threshold=d.get("days_to_threshold"),
                created_at=datetime.utcnow(),
            )
            warnings.append(warning)

        self.logger.info(
            "warnings_computed",
            count=len(warnings),
            metrics=[w.metric for w in warnings],
        )
        return warnings
