"""
Alert Router — Alert Lifecycle Management and Deduplication.

Manages the complete lifecycle of monitor alerts: creation, persistence,
deduplication, acknowledgment, and dismissal. Ensures operational teams
receive actionable, non-duplicated alerts with proper severity routing.

The router prevents alert storms by deduplicating alerts within a
configurable suppression window per monitor, and links alerts to
incidents when the detection engine identifies matching patterns.

Version: alert_router_v1
"""

from datetime import datetime, timedelta
from typing import Optional

import structlog

from api.models.enums import AlertStatus, Severity
from api.models.monitors import MonitorAlert
from api.storage.base import StorageBackend

logger = structlog.get_logger()


# Default suppression window to prevent alert storms
DEFAULT_SUPPRESSION_WINDOW_HOURS = 4

# Severity-based escalation routing
SEVERITY_ROUTING = {
    Severity.CRITICAL: {
        "channels": ["pagerduty", "slack_critical", "email_oncall"],
        "escalation_minutes": 15,
    },
    Severity.HIGH: {
        "channels": ["slack_alerts", "email_oncall"],
        "escalation_minutes": 60,
    },
    Severity.MEDIUM: {
        "channels": ["slack_alerts"],
        "escalation_minutes": 240,
    },
    Severity.LOW: {
        "channels": ["slack_info"],
        "escalation_minutes": None,  # No auto-escalation
    },
}


class AlertRouter:
    """
    Routes, deduplicates, and manages monitor alert lifecycle.

    The AlertRouter is the single entry point for all alerts produced by the
    SLOEvaluator. It ensures deduplication, persists alerts, determines
    routing channels, and tracks acknowledgment state.

    Attributes:
        storage: Storage backend for alert persistence
        suppression_window_hours: Hours within which duplicate alerts are suppressed

    Example:
        >>> router = AlertRouter(storage=duckdb_storage)
        >>> routed = router.route_alerts(alerts)
        >>> for alert, routing in routed:
        ...     print(f"{alert.severity}: {routing['channels']}")
    """

    def __init__(
        self,
        storage: StorageBackend,
        suppression_window_hours: int = DEFAULT_SUPPRESSION_WINDOW_HOURS,
    ):
        """
        Initialize the alert router.

        Args:
            storage: Storage backend for alert persistence
            suppression_window_hours: Window for deduplication in hours
        """
        self.storage = storage
        self.suppression_window_hours = suppression_window_hours
        self.logger = structlog.get_logger()

    def route_alerts(
        self, alerts: list[MonitorAlert]
    ) -> list[tuple[MonitorAlert, dict]]:
        """
        Process, deduplicate, persist, and route a batch of alerts.

        For each alert:
        1. Check if a duplicate exists within the suppression window
        2. If not suppressed, persist the alert
        3. Determine routing based on severity
        4. Return alert + routing information

        Args:
            alerts: List of MonitorAlert instances from SLOEvaluator

        Returns:
            List of (MonitorAlert, routing_dict) tuples for non-suppressed alerts.
            routing_dict contains:
                - channels: list[str] — notification channels
                - escalation_minutes: Optional[int] — auto-escalation timeout
                - suppressed: bool — whether alert was suppressed

        Example:
            >>> results = router.route_alerts(evaluator.evaluate_all_monitors())
        """
        routed: list[tuple[MonitorAlert, dict]] = []

        for alert in alerts:
            try:
                # Check deduplication
                if self._is_suppressed(alert):
                    self.logger.debug(
                        "alert_suppressed",
                        monitor_id=alert.monitor_id,
                        metric_name=alert.metric_name,
                    )
                    routed.append((alert, {"suppressed": True, "channels": [], "escalation_minutes": None}))
                    continue

                # Persist
                self.storage.write_monitor_alert(alert)

                # Determine routing
                routing = self._get_routing(alert.severity)
                routing["suppressed"] = False

                self.logger.info(
                    "alert_routed",
                    alert_id=alert.alert_id,
                    monitor_id=alert.monitor_id,
                    severity=alert.severity.value,
                    channels=routing["channels"],
                )

                routed.append((alert, routing))

            except Exception as e:
                self.logger.error(
                    "alert_routing_failed",
                    alert_id=alert.alert_id,
                    error=str(e),
                )

        self.logger.info(
            "alert_routing_complete",
            total_alerts=len(alerts),
            routed_count=sum(1 for _, r in routed if not r.get("suppressed")),
            suppressed_count=sum(1 for _, r in routed if r.get("suppressed")),
        )

        return routed

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert, marking it as reviewed by a human operator.

        Args:
            alert_id: ID of the alert to acknowledge

        Returns:
            True if alert was found and acknowledged
        """
        alerts = self.storage.read_monitor_alerts(status=AlertStatus.ACTIVE.value)
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                self.storage.write_monitor_alert(alert)
                self.logger.info("alert_acknowledged", alert_id=alert_id)
                return True
        return False

    def dismiss_alert(self, alert_id: str) -> bool:
        """
        Dismiss an alert, marking it as not actionable.

        Args:
            alert_id: ID of the alert to dismiss

        Returns:
            True if alert was found and dismissed
        """
        alerts = self.storage.read_monitor_alerts()
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.DISMISSED
                self.storage.write_monitor_alert(alert)
                self.logger.info("alert_dismissed", alert_id=alert_id)
                return True
        return False

    def link_alert_to_incident(self, alert_id: str, incident_id: str) -> bool:
        """
        Link an alert to a detected incident.

        Called by the detection engine when a monitor alert correlates to
        an incident detection.

        Args:
            alert_id: ID of the alert
            incident_id: ID of the related incident

        Returns:
            True if alert was found and linked
        """
        alerts = self.storage.read_monitor_alerts()
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.related_incident_id = incident_id
                self.storage.write_monitor_alert(alert)
                self.logger.info(
                    "alert_linked_to_incident",
                    alert_id=alert_id,
                    incident_id=incident_id,
                )
                return True
        return False

    def get_active_alerts(
        self, severity: Optional[Severity] = None
    ) -> list[MonitorAlert]:
        """
        Retrieve all active (unacknowledged) alerts.

        Args:
            severity: Optional filter by severity level

        Returns:
            List of active MonitorAlert instances
        """
        alerts = self.storage.read_monitor_alerts(status=AlertStatus.ACTIVE.value)
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _is_suppressed(self, alert: MonitorAlert) -> bool:
        """
        Check if a similar alert was recently fired (deduplication).

        An alert is suppressed if another alert for the same monitor exists
        within the suppression window.

        Args:
            alert: Alert to check

        Returns:
            True if alert should be suppressed
        """
        recent_alerts = self.storage.read_monitor_alerts(
            monitor_id=alert.monitor_id,
        )

        cutoff = datetime.utcnow() - timedelta(hours=self.suppression_window_hours)

        for existing in recent_alerts:
            if (
                existing.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]
                and existing.triggered_at > cutoff
            ):
                return True

        return False

    def _get_routing(self, severity: Severity) -> dict:
        """
        Determine routing channels and escalation policy for a severity level.

        Args:
            severity: Alert severity

        Returns:
            Routing dictionary with channels and escalation info
        """
        routing = SEVERITY_ROUTING.get(severity, SEVERITY_ROUTING[Severity.LOW])
        return {
            "channels": list(routing["channels"]),
            "escalation_minutes": routing["escalation_minutes"],
        }
