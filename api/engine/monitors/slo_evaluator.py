"""
SLO Evaluator — Monitor Rule Condition Evaluation.

Evaluates business metric values against monitor rule thresholds to detect
conditions requiring alerting. Supports baseline-relative conditions (e.g.,
"value > baseline * 2.0") and absolute conditions (e.g., "value < 0.5").

The evaluator computes baselines from Gold layer metrics using the monitor's
configured baseline_window_days, then evaluates each rule's condition expression
against the latest metric value.

Version: slo_eval_v1
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

import structlog

from api.models.enums import Severity
from api.models.monitors import MonitorAlert, MonitorRule
from api.storage.base import StorageBackend

logger = structlog.get_logger()


# Condition expression patterns
_CONDITION_PATTERN = re.compile(
    r"^(value)\s*(>|<|>=|<=|==|!=)\s*(.+)$"
)


class SLOEvaluator:
    """
    Evaluates monitor rule conditions against live metric data.

    For each enabled monitor rule, the evaluator:
    1. Fetches the latest metric value from Gold layer
    2. Computes the baseline from the configured window
    3. Parses and evaluates the condition expression
    4. Returns triggered alerts for conditions that are met

    Attributes:
        storage: Storage backend for metric retrieval
        logger: Structured logger

    Example:
        >>> evaluator = SLOEvaluator(storage=duckdb_storage)
        >>> alerts = evaluator.evaluate_all_monitors()
        >>> for alert in alerts:
        ...     print(f"{alert.metric_name}: {alert.message}")
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize the SLO evaluator.

        Args:
            storage: Storage backend for Gold metric access
        """
        self.storage = storage
        self.logger = structlog.get_logger()

    def evaluate_all_monitors(self) -> list[MonitorAlert]:
        """
        Evaluate all enabled monitor rules and return triggered alerts.

        Fetches all enabled monitors, evaluates each one, and returns a list
        of alerts for rules whose conditions are met.

        Returns:
            List of MonitorAlert instances for triggered monitors

        Example:
            >>> alerts = evaluator.evaluate_all_monitors()
        """
        monitors = self.storage.read_monitors(enabled=True)

        self.logger.info(
            "slo_evaluation_started",
            monitor_count=len(monitors),
        )

        alerts = []
        for monitor in monitors:
            try:
                alert = self.evaluate_monitor(monitor)
                if alert is not None:
                    alerts.append(alert)
            except Exception as e:
                self.logger.error(
                    "monitor_evaluation_failed",
                    monitor_id=monitor.monitor_id,
                    error=str(e),
                )

        self.logger.info(
            "slo_evaluation_complete",
            monitors_evaluated=len(monitors),
            alerts_triggered=len(alerts),
        )

        return alerts

    def evaluate_monitor(self, monitor: MonitorRule) -> Optional[MonitorAlert]:
        """
        Evaluate a single monitor rule.

        Fetches metric data, computes baseline, and evaluates the condition.

        Args:
            monitor: MonitorRule to evaluate

        Returns:
            MonitorAlert if condition is triggered, None otherwise
        """
        # Compute date range for baseline
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (
            datetime.utcnow() - timedelta(days=monitor.baseline_window_days)
        ).strftime("%Y-%m-%d")

        # Fetch metric values from Gold layer
        metrics = self.storage.read_gold_metrics(
            metric_names=[monitor.metric_name],
            start_date=start_date,
            end_date=end_date,
        )

        if not metrics:
            self.logger.debug(
                "no_metric_data",
                monitor_id=monitor.monitor_id,
                metric_name=monitor.metric_name,
            )
            return None

        # Extract values
        values = [m["metric_value"] for m in metrics if "metric_value" in m]
        if not values:
            return None

        current_value = values[-1]  # Most recent
        baseline_value = sum(values) / len(values)  # Mean baseline

        # Evaluate condition
        triggered = self._evaluate_condition(
            condition=monitor.condition,
            value=current_value,
            baseline=baseline_value,
        )

        if not triggered:
            return None

        # Build alert message from template
        message = self._format_alert_message(
            template=monitor.alert_message_template,
            value=current_value,
            baseline=baseline_value,
        )

        alert = MonitorAlert(
            monitor_id=monitor.monitor_id,
            metric_name=monitor.metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            threshold=monitor.condition,
            severity=monitor.severity_if_triggered,
            message=message,
            related_incident_id="",  # Filled by AlertRouter if linked
        )

        self.logger.info(
            "monitor_triggered",
            monitor_id=monitor.monitor_id,
            monitor_name=monitor.name,
            metric_name=monitor.metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            severity=monitor.severity_if_triggered.value,
        )

        return alert

    # =========================================================================
    # Condition Evaluation
    # =========================================================================

    def _evaluate_condition(
        self,
        condition: str,
        value: float,
        baseline: float,
    ) -> bool:
        """
        Parse and evaluate a condition expression.

        Supports expressions like:
        - "value > baseline * 2.0"
        - "value < baseline * 0.5"
        - "value > 100"
        - "value < -0.1"

        Args:
            condition: Condition expression string
            value: Current metric value
            baseline: Computed baseline value

        Returns:
            True if condition is met, False otherwise
        """
        try:
            # Sanitized evaluation with restricted namespace
            safe_namespace = {
                "value": value,
                "baseline": baseline,
                "abs": abs,
                "min": min,
                "max": max,
            }
            result = eval(condition, {"__builtins__": {}}, safe_namespace)  # noqa: S307
            return bool(result)
        except Exception as e:
            self.logger.warning(
                "condition_evaluation_failed",
                condition=condition,
                value=value,
                baseline=baseline,
                error=str(e),
            )
            return False

    def _format_alert_message(
        self,
        template: str,
        value: float,
        baseline: float,
    ) -> str:
        """
        Format alert message template with current values.

        Supports placeholders: {value}, {baseline}, {multiplier}

        Args:
            template: Message template string
            value: Current metric value
            baseline: Baseline value

        Returns:
            Formatted message string
        """
        try:
            multiplier = value / baseline if baseline != 0 else float("inf")
            return template.format(
                value=value,
                baseline=baseline,
                multiplier=multiplier,
            )
        except (KeyError, ValueError, ZeroDivisionError):
            return (
                f"Monitor alert triggered: value={value:.4f}, "
                f"baseline={baseline:.4f}"
            )
