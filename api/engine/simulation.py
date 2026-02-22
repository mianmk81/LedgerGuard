"""
What-If Simulation Engine (Component K).

Lets admins ask "what would happen if [metric] changed by X%?" and see
which incidents would trigger. Uses the Business Dependency Graph to
propagate perturbations downstream.

This module implements:
- Metric perturbation propagation through dependency graphs
- Historical correlation estimation for impact prediction
- Threshold-based incident trigger prediction
- Cascade correlation for multi-hop impacts
- Natural language narrative generation

Note: Uses simple linear estimates from historical correlation.
This is a stress test tool, not a prediction engine.

Example usage:
    >>> simulator = WhatIfSimulator(storage)
    >>> scenario = simulator.simulate(
    ...     perturbations=[{"metric": "delivery_delay_rate", "change_pct": 30}],
    ... )
    >>> print(scenario.narrative)
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import numpy as np
import structlog
from scipy import stats

from api.models.simulation import WhatIfScenario
from api.storage.base import StorageBackend

logger = structlog.get_logger()


# =========================================================================
# Business Dependency Graph
# =========================================================================


class BusinessDependencyGraph:
    """
    Business metric dependency graph for impact propagation.

    Constructs a directed acyclic graph (DAG) of metric dependencies based on
    historical correlation analysis and domain knowledge. Used by the simulation
    engine to propagate perturbations through the business system.

    The graph encodes relationships like:
    - delivery_delay_rate → refund_rate (operational → financial)
    - refund_rate → support_load (financial → customer)
    - support_load → customer_satisfaction (customer → customer)

    Attributes:
        graph: NetworkX DiGraph representing metric dependencies
        logger: Structured logger for observability
    """

    def __init__(self):
        """Initialize the dependency graph with domain-knowledge edges."""
        self.graph = nx.DiGraph()
        self.logger = structlog.get_logger()
        self._build_default_graph()

    def _build_default_graph(self) -> None:
        """
        Build default dependency graph based on domain knowledge.

        This is a simplified version. In production, this would be learned
        from historical data using Granger causality, transfer entropy, or
        other causal discovery methods.
        """
        # Operational → Financial edges
        self.graph.add_edge(
            "delivery_delay_rate",
            "refund_rate",
            correlation=0.65,
            description="Late deliveries drive refunds",
        )
        self.graph.add_edge(
            "fulfillment_backlog",
            "delivery_delay_rate",
            correlation=0.72,
            description="Backlog causes delays",
        )
        self.graph.add_edge(
            "supplier_delay_rate",
            "delivery_delay_rate",
            correlation=0.58,
            description="Supplier delays cascade to delivery",
        )

        # Financial → Customer edges
        self.graph.add_edge(
            "refund_rate",
            "ticket_volume",
            correlation=0.78,
            description="Refunds generate support tickets",
        )
        self.graph.add_edge(
            "refund_rate",
            "review_score_avg",
            correlation=-0.55,
            description="Refunds hurt review scores",
        )

        # Operational → Customer edges
        self.graph.add_edge(
            "delivery_delay_rate",
            "ticket_volume",
            correlation=0.68,
            description="Delays generate support requests",
        )
        self.graph.add_edge(
            "delivery_delay_rate",
            "review_score_avg",
            correlation=-0.62,
            description="Delays hurt customer satisfaction",
        )

        # Customer → Customer edges
        self.graph.add_edge(
            "ticket_volume",
            "ticket_backlog",
            correlation=0.85,
            description="Volume drives backlog",
        )
        self.graph.add_edge(
            "ticket_backlog",
            "avg_resolution_time",
            correlation=0.77,
            description="Backlog increases resolution time",
        )
        self.graph.add_edge(
            "review_score_avg",
            "churn_proxy",
            correlation=-0.48,
            description="Poor reviews drive churn",
        )
        self.graph.add_edge(
            "avg_resolution_time",
            "review_score_avg",
            correlation=-0.52,
            description="Slow support hurts satisfaction",
        )

        # Financial → Financial edges
        self.graph.add_edge(
            "refund_rate",
            "margin_proxy",
            correlation=-0.73,
            description="Refunds compress margins",
        )
        self.graph.add_edge(
            "expense_ratio",
            "margin_proxy",
            correlation=-0.88,
            description="Expense ratio inversely affects margin",
        )

        # Customer → Financial edges
        self.graph.add_edge(
            "churn_proxy",
            "daily_revenue",
            correlation=-0.42,
            description="Churn reduces revenue",
        )

        self.logger.info(
            "dependency_graph_built",
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
        )

    def get_downstream_metrics(self, metric: str) -> list[tuple[str, float]]:
        """
        Get all metrics downstream of the given metric with correlation weights.

        Args:
            metric: Metric name to find downstream dependencies for

        Returns:
            List of (downstream_metric, correlation_coefficient) tuples
        """
        if metric not in self.graph:
            return []

        downstream = []
        for successor in self.graph.successors(metric):
            edge_data = self.graph[metric][successor]
            correlation = edge_data.get("correlation", 0.5)
            downstream.append((successor, correlation))

        return downstream

    def get_all_downstream_paths(self, metric: str, max_depth: int = 3) -> list[list[str]]:
        """
        Get all downstream paths from a metric up to max_depth hops.

        Args:
            metric: Starting metric
            max_depth: Maximum path depth to explore

        Returns:
            List of paths, where each path is a list of metric names
        """
        if metric not in self.graph:
            return []

        paths = []

        # BFS to find all paths
        queue = deque([(metric, [metric], 0)])

        while queue:
            current, path, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for successor in self.graph.successors(current):
                new_path = path + [successor]
                paths.append(new_path)
                queue.append((successor, new_path, depth + 1))

        return paths


# =========================================================================
# What-If Simulator
# =========================================================================


class WhatIfSimulator:
    """
    Simulates metric perturbations and predicts incident triggers.

    Uses the Business Dependency Graph to propagate perturbations downstream
    and estimates which incidents would be triggered based on historical
    thresholds and correlation patterns.

    This is a stress testing and capacity planning tool, not a prediction engine.
    Results are approximate and should be validated against historical data.

    Attributes:
        storage: Storage backend for reading metrics and thresholds
        dep_graph: Business dependency graph for propagation
        logger: Structured logger for observability
    """

    # Attenuation factor per hop (diminishes with distance)
    ATTENUATION_FACTOR = 0.7

    # Incident detection thresholds (simplified)
    INCIDENT_THRESHOLDS = {
        "refund_spike": {"metric": "refund_rate", "threshold": 0.08, "severity": "high"},
        "fulfillment_sla_degradation": {
            "metric": "delivery_delay_rate",
            "threshold": 0.20,
            "severity": "high",
        },
        "support_load_surge": {
            "metric": "ticket_backlog",
            "threshold": 40,
            "severity": "medium",
        },
        "customer_satisfaction_regression": {
            "metric": "review_score_avg",
            "threshold": 3.0,
            "below": True,
            "severity": "medium",
        },
        "margin_compression": {
            "metric": "margin_proxy",
            "threshold": 0.15,
            "below": True,
            "severity": "high",
        },
        "churn_acceleration": {
            "metric": "churn_proxy",
            "threshold": 0.05,
            "severity": "critical",
        },
    }

    def __init__(
        self,
        storage: StorageBackend,
        dependency_graph: Optional[BusinessDependencyGraph] = None,
    ):
        """
        Initialize the what-if simulator.

        Args:
            storage: Storage backend for reading metrics
            dependency_graph: Optional custom dependency graph (uses default if None)
        """
        self.storage = storage
        self.dep_graph = dependency_graph or BusinessDependencyGraph()
        self.logger = structlog.get_logger()

    def simulate(
        self,
        perturbations: list[dict],
        current_metrics: Optional[dict] = None,
        historical_metrics: Optional[list[dict]] = None,
    ) -> WhatIfScenario:
        """
        Run a what-if simulation.

        Args:
            perturbations: List of perturbation dicts with "metric" and "change_pct" keys
                Example: [{"metric": "delivery_delay_rate", "change_pct": 30}]
            current_metrics: Current Gold metrics snapshot (fetched if None)
            historical_metrics: Historical metrics for correlation estimation (fetched if None)

        Returns:
            WhatIfScenario with simulated metrics, triggered incidents, cascades, narrative

        Example:
            >>> scenario = simulator.simulate(
            ...     perturbations=[{"metric": "order_volume", "change_pct": 50}],
            ... )
            >>> print(f"Would trigger: {scenario.triggered_incidents}")
        """
        self.logger.info(
            "simulation_started",
            perturbations=perturbations,
        )

        # Normalize perturbations: accept "change" (e.g. "+50%") or "change_pct" (e.g. 50)
        perturbations = self._normalize_perturbations(perturbations)

        # Step 1: Get current baseline metrics
        if current_metrics is None:
            current_metrics = self._get_current_metrics()

        # Step 2: Get historical metrics for correlation estimation
        if historical_metrics is None:
            historical_metrics = self._get_historical_metrics(days=30)

        # Step 3: Apply direct perturbations
        simulated_metrics = self._apply_perturbations(current_metrics, perturbations)

        # Step 4: Propagate through dependency graph
        simulated_metrics = self._propagate_through_graph(
            current_metrics, simulated_metrics, historical_metrics
        )

        # Step 5: Check thresholds to predict triggered incidents
        triggered_incidents_detail = self._check_thresholds(
            simulated_metrics, current_metrics
        )
        triggered_incidents = [inc["type"] for inc in triggered_incidents_detail]

        # Step 5b: Run ML models on simulated metrics for richer predictions
        ml_insights = self._run_ml_predictions(simulated_metrics, current_metrics)

        # Step 6: Identify cascade patterns
        triggered_cascades = self._identify_cascades(triggered_incidents_detail)

        # Step 7: Generate narrative
        narrative = self._generate_narrative(
            perturbations, triggered_incidents_detail, triggered_cascades,
            current_metrics, ml_insights,
        )

        # Build scenario
        scenario = WhatIfScenario(
            perturbations=[
                {"metric": p["metric"], "change": f"{p['change_pct']:+.1f}%"}
                for p in perturbations
            ],
            simulated_metrics=simulated_metrics,
            triggered_incidents=triggered_incidents,
            triggered_cascades=triggered_cascades,
            narrative=narrative,
            ml_insights=ml_insights,
            models_used=ml_insights.get("models_used", []),
        )

        self.logger.info(
            "simulation_completed",
            scenario_id=scenario.scenario_id,
            triggered_incidents_count=len(triggered_incidents),
            triggered_cascades_count=len(triggered_cascades),
        )

        return scenario

    def _normalize_perturbations(self, perturbations: list[dict]) -> list[dict]:
        """Convert 'change' (e.g. '+50%') to 'change_pct' (50.0) for compatibility."""
        import re
        result = []
        for p in perturbations:
            p = dict(p)
            if "change_pct" in p:
                result.append(p)
                continue
            if "change" in p:
                s = str(p["change"]).strip()
                m = re.match(r"([+-]?\d*\.?\d+)\s*%?", s)
                if m:
                    p["change_pct"] = float(m.group(1))
                else:
                    p["change_pct"] = 0.0
            else:
                p["change_pct"] = 0.0
            result.append(p)
        return result

    def _get_current_metrics(self) -> dict:
        """
        Fetch current Gold layer metrics snapshot.

        Returns:
            Dictionary of current metric values
        """
        # Try to get yesterday's metrics (most recent complete day)
        from datetime import date

        yesterday = str(date.today() - timedelta(days=1))

        stored_metrics = self.storage.read_gold_metrics(
            metric_names=None,
            start_date=yesterday,
            end_date=yesterday,
        )

        if not stored_metrics:
            self.logger.warning("no_current_metrics_found", using_defaults=True)
            # Return reasonable defaults
            return self._get_default_metrics()

        # Flatten stored metrics into dict
        metrics = {}
        for metric in stored_metrics:
            metrics[metric["metric_name"]] = metric["metric_value"]

        return metrics

    def _get_historical_metrics(self, days: int = 30) -> list[dict]:
        """
        Fetch historical Gold layer metrics for correlation estimation.

        Args:
            days: Number of days of history to fetch

        Returns:
            List of daily metric snapshots
        """
        from datetime import date

        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)

        stored_metrics = self.storage.read_gold_metrics(
            metric_names=None,
            start_date=str(start_date),
            end_date=str(end_date),
        )

        # Group by date
        by_date = {}
        for metric in stored_metrics:
            metric_date = metric["metric_date"]
            if metric_date not in by_date:
                by_date[metric_date] = {}
            by_date[metric_date][metric["metric_name"]] = metric["metric_value"]

        # Convert to list of dicts
        historical = [
            {"date": date_str, **metrics} for date_str, metrics in sorted(by_date.items())
        ]

        return historical

    def _get_default_metrics(self) -> dict:
        """
        Return default baseline metrics when no data available.

        Returns:
            Dictionary of default metric values
        """
        return {
            # Financial
            "daily_revenue": 10000.0,
            "daily_expenses": 7000.0,
            "daily_refunds": 300.0,
            "refund_rate": 0.03,
            "margin_proxy": 0.27,
            "expense_ratio": 0.70,
            # Operational
            "order_volume": 100,
            "delivery_count": 95,
            "delivery_delay_rate": 0.08,
            "fulfillment_backlog": 25,
            "supplier_delay_rate": 0.10,
            # Customer
            "ticket_volume": 20,
            "ticket_backlog": 15,
            "avg_resolution_time": 4.0,
            "review_score_avg": 4.2,
            "churn_proxy": 0.02,
        }

    def _apply_perturbations(
        self, current_metrics: dict, perturbations: list[dict]
    ) -> dict:
        """
        Apply direct perturbations to specified metrics.

        Args:
            current_metrics: Current baseline metric values
            perturbations: List of perturbation specs

        Returns:
            Dictionary with perturbed metric values
        """
        simulated = current_metrics.copy()

        for perturbation in perturbations:
            metric = perturbation["metric"]
            change_pct = perturbation["change_pct"]

            if metric in simulated:
                current_value = simulated[metric]
                new_value = current_value * (1 + change_pct / 100.0)
                simulated[metric] = new_value

                self.logger.info(
                    "perturbation_applied",
                    metric=metric,
                    change_pct=change_pct,
                    current_value=current_value,
                    new_value=new_value,
                )

        return simulated

    def _propagate_through_graph(
        self,
        current_metrics: dict,
        perturbed_metrics: dict,
        historical_metrics: list[dict],
    ) -> dict:
        """
        Propagate perturbations through dependency graph.

        Uses BFS to propagate changes downstream with attenuation per hop.
        Correlation coefficients from the graph are used to scale impact.

        Args:
            current_metrics: Original baseline metrics
            perturbed_metrics: Metrics after direct perturbations
            historical_metrics: Historical data for correlation validation

        Returns:
            Fully propagated metric values
        """
        simulated = perturbed_metrics.copy()

        # Identify which metrics were perturbed
        perturbed_list = [
            metric
            for metric in simulated
            if abs(simulated[metric] - current_metrics.get(metric, 0)) > 1e-6
        ]

        # BFS propagation
        queue = deque([(metric, 0) for metric in perturbed_list])
        visited = set()

        while queue:
            metric, depth = queue.popleft()

            if metric in visited or depth >= 3:  # Max 3 hops
                continue

            visited.add(metric)

            # Get downstream metrics
            downstream = self.dep_graph.get_downstream_metrics(metric)

            for downstream_metric, correlation in downstream:
                if downstream_metric not in simulated:
                    continue

                # Calculate change in upstream metric
                current_value = current_metrics.get(metric, 1.0)
                perturbed_value = simulated[metric]

                if current_value == 0:
                    continue

                change_ratio = (perturbed_value - current_value) / current_value

                # Propagate with correlation and attenuation
                attenuation = self.ATTENUATION_FACTOR ** (depth + 1)
                propagated_change = change_ratio * correlation * attenuation

                # Apply to downstream metric
                downstream_current = current_metrics.get(downstream_metric, 1.0)
                downstream_new = downstream_current * (1 + propagated_change)

                # Update if impact is significant
                if abs(propagated_change) > 0.01:  # 1% threshold
                    simulated[downstream_metric] = downstream_new

                    self.logger.info(
                        "propagation_applied",
                        from_metric=metric,
                        to_metric=downstream_metric,
                        correlation=correlation,
                        propagated_change_pct=propagated_change * 100,
                        depth=depth + 1,
                    )

                    # Add to queue for further propagation
                    queue.append((downstream_metric, depth + 1))

        return simulated

    def _estimate_correlation(
        self, metric_a_values: list[float], metric_b_values: list[float]
    ) -> float:
        """
        Estimate Pearson correlation between two metric time series.

        Args:
            metric_a_values: Time series values for metric A
            metric_b_values: Time series values for metric B

        Returns:
            Pearson correlation coefficient (-1 to 1)
        """
        if len(metric_a_values) < 3 or len(metric_b_values) < 3:
            return 0.0

        if len(metric_a_values) != len(metric_b_values):
            return 0.0

        try:
            correlation, _ = stats.pearsonr(metric_a_values, metric_b_values)
            return correlation
        except Exception as e:
            self.logger.warning(
                "correlation_estimation_failed",
                error=str(e),
            )
            return 0.0

    def _run_ml_predictions(self, simulated_metrics: dict, current_metrics: dict) -> dict:
        """
        Run ML models on simulated metrics for richer predictions.
        Returns a dict of model results; gracefully falls back if models unavailable.
        """
        insights = {"churn_risk": None, "anomaly_detected": None, "health_score_impact": None, "models_used": []}

        # Churn prediction on simulated customer metrics
        try:
            from api.engine.prediction.churn_predictor import predict_churn_risk
            churn_features = {
                "recency_days": 30,
                "frequency": int(simulated_metrics.get("order_volume", 0) * 30),
                "monetary": simulated_metrics.get("daily_revenue", 0) * 30,
                "avg_review_score": simulated_metrics.get("review_score_avg", 4.0),
                "late_order_rate": simulated_metrics.get("delivery_delay_rate", 0),
                "complaint_count": int(simulated_metrics.get("ticket_volume", 0)),
            }
            result = predict_churn_risk(churn_features)
            if result.get("model_used"):
                insights["churn_risk"] = result.get("probability", 0)
                insights["models_used"].append("Churn (LightGBM)")
        except Exception:
            pass

        # Health score impact estimate
        try:
            from api.engine.monitors import HealthScorer
            scorer = HealthScorer(storage=self.storage, lookback_days=7)
            current_health = scorer.compute_health()
            current_score = current_health.get("overall_score", 50)

            # Estimate simulated health by approximating metric impact
            sim_score = current_score
            for metric_name in ["refund_rate", "delivery_delay_rate", "churn_proxy"]:
                curr_val = current_metrics.get(metric_name, 0)
                sim_val = simulated_metrics.get(metric_name, curr_val)
                if curr_val > 0 and sim_val > curr_val:
                    degradation = min(20, (sim_val - curr_val) / curr_val * 30)
                    sim_score -= degradation
            for metric_name in ["margin_proxy", "review_score_avg"]:
                curr_val = current_metrics.get(metric_name, 1)
                sim_val = simulated_metrics.get(metric_name, curr_val)
                if curr_val > 0 and sim_val < curr_val:
                    degradation = min(15, (curr_val - sim_val) / curr_val * 25)
                    sim_score -= degradation

            insights["health_score_impact"] = {
                "current": round(current_score, 1),
                "projected": round(max(0, min(100, sim_score)), 1),
            }
        except Exception:
            pass

        return insights

    def _check_thresholds(
        self, simulated_metrics: dict, current_metrics: dict
    ) -> list[dict]:
        """
        Check simulated metrics against incident thresholds.

        Args:
            simulated_metrics: Simulated metric values
            current_metrics: Current baseline metric values

        Returns:
            List of triggered incident detail dicts
        """
        triggered = []

        for incident_type, config in self.INCIDENT_THRESHOLDS.items():
            metric = config["metric"]
            threshold = config["threshold"]
            severity = config["severity"]
            below = config.get("below", False)

            if metric not in simulated_metrics:
                continue

            simulated_value = simulated_metrics[metric]
            current_value = current_metrics.get(metric, 0)

            # Check threshold
            triggered_flag = False
            if below:
                triggered_flag = simulated_value < threshold
            else:
                triggered_flag = simulated_value > threshold

            if triggered_flag:
                triggered.append(
                    {
                        "type": incident_type,
                        "metric": metric,
                        "simulated_value": simulated_value,
                        "current_value": current_value,
                        "threshold": threshold,
                        "severity": severity,
                        "below": below,
                    }
                )

                self.logger.info(
                    "incident_triggered_in_simulation",
                    incident_type=incident_type,
                    metric=metric,
                    simulated_value=simulated_value,
                    threshold=threshold,
                )

        return triggered

    def _identify_cascades(self, triggered_incidents: list[dict]) -> list[str]:
        """
        Identify cascade patterns in triggered incidents.

        Args:
            triggered_incidents: List of triggered incident detail dicts

        Returns:
            List of cascade description strings
        """
        if len(triggered_incidents) < 2:
            return []

        cascades = []

        # Look for known cascade patterns
        incident_types = [inc["type"] for inc in triggered_incidents]

        # Refund → Support cascade
        if "refund_spike" in incident_types and "support_load_surge" in incident_types:
            cascades.append("refund_spike -> support_load_surge")

        # Fulfillment → Satisfaction cascade
        if (
            "fulfillment_sla_degradation" in incident_types
            and "customer_satisfaction_regression" in incident_types
        ):
            cascades.append("fulfillment_sla_degradation -> customer_satisfaction_regression")

        # Support → Churn cascade
        if "support_load_surge" in incident_types and "churn_acceleration" in incident_types:
            cascades.append("support_load_surge -> churn_acceleration")

        # Satisfaction → Churn cascade
        if (
            "customer_satisfaction_regression" in incident_types
            and "churn_acceleration" in incident_types
        ):
            cascades.append("customer_satisfaction_regression -> churn_acceleration")

        return cascades

    def _generate_narrative(
        self,
        perturbations: list[dict],
        triggered_incidents: list[dict],
        triggered_cascades: list[str],
        current_metrics: dict,
        ml_insights: dict | None = None,
    ) -> str:
        """Generate a plain-English narrative of the simulation results."""
        ml_insights = ml_insights or {}

        metric_labels = {
            "order_volume": "order volume",
            "delivery_delay_rate": "delivery delay rate",
            "refund_rate": "refund rate",
            "margin_proxy": "profit margin",
            "daily_revenue": "daily revenue",
            "daily_expenses": "daily expenses",
            "ticket_volume": "support ticket volume",
            "review_score_avg": "customer review scores",
            "churn_proxy": "customer churn risk",
            "fulfillment_backlog": "fulfillment backlog",
            "supplier_delay_rate": "supplier delay rate",
        }

        pert_parts = []
        for pert in perturbations:
            label = metric_labels.get(pert["metric"], pert["metric"].replace("_", " "))
            change = pert["change_pct"]
            if change > 0:
                pert_parts.append(f"{label} went up by {abs(change):.0f}%")
            else:
                pert_parts.append(f"{label} dropped by {abs(change):.0f}%")

        scenario_text = " and ".join(pert_parts) if pert_parts else "no changes"

        if not triggered_incidents:
            narrative = (
                f"If {scenario_text}, your business would stay within safe limits. "
                f"No incidents would be triggered."
            )
        else:
            problem_parts = []
            for inc in triggered_incidents:
                label = metric_labels.get(inc["metric"], inc["metric"].replace("_", " "))
                severity = inc["severity"]
                severity_word = "serious" if severity in ("critical", "high") else "moderate"
                problem_parts.append(f"a {severity_word} {inc['type'].replace('_', ' ')}")

            problems_text = ", ".join(problem_parts)
            time_est = self._estimate_time_to_impact(perturbations[0]["metric"])

            narrative = (
                f"If {scenario_text}, it would likely cause {problems_text} "
                f"within {time_est}."
            )

            if triggered_cascades:
                cascade_labels = []
                for c in triggered_cascades:
                    parts = c.split(" -> ")
                    readable = " leading to ".join(
                        p.replace("_", " ") for p in parts
                    )
                    cascade_labels.append(readable)
                narrative += f" These problems would cascade: {'; '.join(cascade_labels)}."

        # Add ML model insights
        health_impact = ml_insights.get("health_score_impact")
        if health_impact:
            curr = health_impact["current"]
            proj = health_impact["projected"]
            diff = curr - proj
            if diff > 5:
                narrative += (
                    f" Your health score would drop from {curr:.0f} to {proj:.0f} "
                    f"(a {diff:.0f}-point decline)."
                )

        churn_risk = ml_insights.get("churn_risk")
        if churn_risk is not None and churn_risk > 0.1:
            narrative += (
                f" Our churn model predicts {churn_risk:.0%} of customers "
                f"would be at risk of leaving under this scenario."
            )

        models_used = ml_insights.get("models_used", [])
        if models_used:
            narrative += f" (Powered by: {', '.join(models_used)})"

        return narrative

    def _estimate_time_to_impact(self, perturbation_metric: str) -> str:
        """
        Estimate time for perturbation to manifest as incidents.

        Args:
            perturbation_metric: Metric being perturbed

        Returns:
            Time estimate string
        """
        # Simplified time estimates based on metric domain
        operational_metrics = [
            "order_volume",
            "delivery_delay_rate",
            "fulfillment_backlog",
        ]
        financial_metrics = ["daily_revenue", "refund_rate", "margin_proxy"]
        customer_metrics = ["ticket_volume", "review_score_avg", "churn_proxy"]

        if perturbation_metric in operational_metrics:
            return "1-3 days"
        elif perturbation_metric in financial_metrics:
            return "2-5 days"
        elif perturbation_metric in customer_metrics:
            return "3-7 days"
        else:
            return "3-5 days"


# Alias for API router imports
SimulationEngine = WhatIfSimulator
