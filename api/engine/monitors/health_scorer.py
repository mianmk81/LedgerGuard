"""
Health Scorer — Composite Business Health Assessment.

Computes a multi-domain composite health score for the business by evaluating
Gold layer metrics across Financial, Operational, and Customer domains. Each
domain produces a 0-100 score, which is weighted into an overall Business
Reliability Score (BRS).

Domain scoring uses configurable thresholds to grade metrics from "healthy"
(100) through "degraded" (50) to "critical" (0), with sigmoid smoothing.

Version: health_scorer_v1
"""

import math
from datetime import datetime, timedelta
from typing import Optional

import structlog

from api.storage.base import StorageBackend

logger = structlog.get_logger()


# ============================================================================
# Domain Definitions — metrics, weights, and thresholds
# ============================================================================

FINANCIAL_METRICS = {
    "margin_proxy": {
        "weight": 0.25,
        "healthy_min": 0.15,      # ≥15% margin = healthy
        "critical_below": 0.05,   # <5% margin = critical
        "direction": "higher_better",
    },
    "refund_rate": {
        "weight": 0.20,
        "healthy_max": 0.03,      # ≤3% = healthy
        "critical_above": 0.10,   # >10% = critical
        "direction": "lower_better",
    },
    "net_cash_proxy": {
        "weight": 0.20,
        "healthy_min": 0,
        "critical_below": -10000,
        "direction": "higher_better",
    },
    "expense_ratio": {
        "weight": 0.15,
        "healthy_max": 0.70,      # ≤70% expense ratio = healthy
        "critical_above": 0.95,   # >95% = critical
        "direction": "lower_better",
    },
    "dso_proxy": {
        "weight": 0.20,
        "healthy_max": 30,        # ≤30 days = healthy
        "critical_above": 90,     # >90 days = critical
        "direction": "lower_better",
    },
}

OPERATIONAL_METRICS = {
    "delivery_delay_rate": {
        "weight": 0.25,
        "healthy_max": 0.05,
        "critical_above": 0.20,
        "direction": "lower_better",
    },
    "fulfillment_backlog": {
        "weight": 0.20,
        "healthy_max": 50,
        "critical_above": 200,
        "direction": "lower_better",
    },
    "supplier_delay_rate": {
        "weight": 0.20,
        "healthy_max": 0.05,
        "critical_above": 0.20,
        "direction": "lower_better",
    },
    "order_volume": {
        "weight": 0.15,
        "healthy_min": 10,
        "critical_below": 1,
        "direction": "higher_better",
    },
    "avg_delivery_delay_days": {
        "weight": 0.20,
        "healthy_max": 2,
        "critical_above": 10,
        "direction": "lower_better",
    },
}

CUSTOMER_METRICS = {
    "review_score_avg": {
        "weight": 0.25,
        "healthy_min": 4.0,
        "critical_below": 2.5,
        "direction": "higher_better",
    },
    "ticket_backlog": {
        "weight": 0.20,
        "healthy_max": 20,
        "critical_above": 100,
        "direction": "lower_better",
    },
    "churn_proxy": {
        "weight": 0.25,
        "healthy_max": 0.02,
        "critical_above": 0.10,
        "direction": "lower_better",
    },
    "avg_resolution_time": {
        "weight": 0.15,
        "healthy_max": 24,        # hours
        "critical_above": 96,
        "direction": "lower_better",
    },
    "review_score_trend": {
        "weight": 0.15,
        "healthy_min": -0.02,
        "critical_below": -0.20,
        "direction": "higher_better",
    },
}

# Overall domain weights for Business Reliability Score
DOMAIN_WEIGHTS = {
    "financial": 0.40,
    "operational": 0.35,
    "customer": 0.25,
}


class HealthScorer:
    """
    Computes composite business health scores across multiple domains.

    The HealthScorer evaluates the latest Gold metrics against domain-specific
    thresholds to produce a 0-100 Business Reliability Score (BRS) with
    per-domain breakdowns.

    Attributes:
        storage: Storage backend for Gold metric retrieval
        lookback_days: Number of days to consider for latest metrics

    Example:
        >>> scorer = HealthScorer(storage=duckdb_storage)
        >>> health = scorer.compute_health()
        >>> print(f"BRS: {health['overall_score']}/100")
        >>> print(f"Financial: {health['domains']['financial']['score']}")
    """

    def __init__(self, storage: StorageBackend, lookback_days: int = 7):
        """
        Initialize the health scorer.

        Args:
            storage: Storage backend for Gold metric access
            lookback_days: Number of recent days to pull metrics from
        """
        self.storage = storage
        self.lookback_days = lookback_days
        self.logger = structlog.get_logger()

    def compute_health(self) -> dict:
        """
        Compute overall Business Reliability Score and domain breakdowns.

        Returns a comprehensive health report with:
        - overall_score: 0-100 composite BRS
        - overall_grade: A/B/C/D/F letter grade
        - domains: per-domain scores and metric details
        - evaluated_at: timestamp of evaluation
        - trend: comparison to previous period (if available)

        Returns:
            dict with health assessment results

        Example:
            >>> health = scorer.compute_health()
            >>> print(health["overall_grade"])  # "B"
        """
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (
            datetime.utcnow() - timedelta(days=self.lookback_days)
        ).strftime("%Y-%m-%d")

        # Fetch all relevant metrics
        all_metric_names = (
            list(FINANCIAL_METRICS.keys())
            + list(OPERATIONAL_METRICS.keys())
            + list(CUSTOMER_METRICS.keys())
        )

        metrics = self.storage.read_gold_metrics(
            metric_names=all_metric_names,
            start_date=start_date,
            end_date=end_date,
        )

        # Build metric value map (latest value per metric)
        metric_map = self._build_latest_metric_map(metrics)

        # Score each domain
        financial_result = self._score_domain(
            "financial", FINANCIAL_METRICS, metric_map
        )
        operational_result = self._score_domain(
            "operational", OPERATIONAL_METRICS, metric_map
        )
        customer_result = self._score_domain(
            "customer", CUSTOMER_METRICS, metric_map
        )

        # Compute weighted overall score
        overall_score = (
            financial_result["score"] * DOMAIN_WEIGHTS["financial"]
            + operational_result["score"] * DOMAIN_WEIGHTS["operational"]
            + customer_result["score"] * DOMAIN_WEIGHTS["customer"]
        )

        overall_grade = self._score_to_grade(overall_score)

        # Compute trend vs previous period
        trend = self._compute_trend(all_metric_names, start_date, current_score=overall_score)

        explanation = self._generate_plain_explanation(
            overall_score, overall_grade, financial_result, operational_result, customer_result
        )

        result = {
            "overall_score": round(overall_score, 1),
            "overall_grade": overall_grade,
            "explanation": explanation,
            "domains": {
                "financial": financial_result,
                "operational": operational_result,
                "customer": customer_result,
            },
            "evaluated_at": datetime.utcnow().isoformat(),
            "lookback_days": self.lookback_days,
            "metrics_available": len(metric_map),
            "metrics_expected": len(all_metric_names),
            "trend": trend,
        }

        self.logger.info(
            "health_score_computed",
            overall_score=result["overall_score"],
            overall_grade=overall_grade,
            financial=financial_result["score"],
            operational=operational_result["score"],
            customer=customer_result["score"],
        )

        return result

    def compute_domain_health(self, domain: str) -> dict:
        """
        Compute health for a single domain.

        Args:
            domain: One of "financial", "operational", "customer"

        Returns:
            Domain health result dict

        Raises:
            ValueError: If domain name is invalid
        """
        domain_configs = {
            "financial": FINANCIAL_METRICS,
            "operational": OPERATIONAL_METRICS,
            "customer": CUSTOMER_METRICS,
        }

        if domain not in domain_configs:
            raise ValueError(
                f"Invalid domain '{domain}'. Must be one of: {list(domain_configs.keys())}"
            )

        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (
            datetime.utcnow() - timedelta(days=self.lookback_days)
        ).strftime("%Y-%m-%d")

        config = domain_configs[domain]
        metric_names = list(config.keys())

        metrics = self.storage.read_gold_metrics(
            metric_names=metric_names,
            start_date=start_date,
            end_date=end_date,
        )

        metric_map = self._build_latest_metric_map(metrics)
        return self._score_domain(domain, config, metric_map)

    # =========================================================================
    # Scoring Logic
    # =========================================================================

    def _score_domain(
        self, domain_name: str, metric_configs: dict, metric_map: dict
    ) -> dict:
        """
        Score a single domain against its metric thresholds.

        Args:
            domain_name: Name of the domain
            metric_configs: Dict of metric name → config
            metric_map: Dict of metric name → latest value

        Returns:
            Dict with domain score (0-100), grade, and per-metric details
        """
        metric_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for metric_name, config in metric_configs.items():
            value = metric_map.get(metric_name)

            if value is None:
                # Missing metric — neutral score, reduced weight
                metric_scores[metric_name] = {
                    "value": None,
                    "score": None,
                    "status": "no_data",
                    "weight": config["weight"],
                }
                continue

            score = self._score_metric(value, config)
            status = self._score_to_status(score)

            metric_scores[metric_name] = {
                "value": round(value, 4),
                "score": round(score, 1),
                "status": status,
                "weight": config["weight"],
            }

            weighted_sum += score * config["weight"]
            total_weight += config["weight"]

        # Normalize for missing metrics
        domain_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 50.0

        grade = self._score_to_grade(domain_score)

        # Find metrics dragging the score down
        weak_metrics = [
            name.replace("_", " ")
            for name, data in metric_scores.items()
            if data.get("score") is not None and data["score"] < 50
        ]
        domain_explanation = ""
        if grade in ("A", "B"):
            domain_explanation = f"Your {domain_name} health is strong"
        elif grade == "C":
            domain_explanation = f"Your {domain_name} health needs attention"
        else:
            domain_explanation = f"Your {domain_name} health is in poor shape"

        if weak_metrics:
            domain_explanation += f" — {', '.join(weak_metrics[:3])} {'is' if len(weak_metrics) == 1 else 'are'} the main concern{'s' if len(weak_metrics) > 1 else ''}"

        return {
            "score": round(domain_score, 1),
            "grade": grade,
            "explanation": domain_explanation,
            "metrics": metric_scores,
            "metrics_available": sum(
                1 for m in metric_scores.values() if m["value"] is not None
            ),
            "metrics_total": len(metric_configs),
        }

    def _score_metric(self, value: float, config: dict) -> float:
        """
        Score a single metric value against its threshold config.

        Uses sigmoid smoothing for continuous scoring between 0 and 1.

        Args:
            value: Current metric value
            config: Metric threshold configuration

        Returns:
            Score between 0.0 (critical) and 1.0 (healthy)
        """
        direction = config.get("direction", "lower_better")

        if direction == "lower_better":
            healthy_max = config.get("healthy_max", 0)
            critical_above = config.get("critical_above", healthy_max * 3)

            if value <= healthy_max:
                return 1.0
            elif value >= critical_above:
                return 0.0
            else:
                # Linear interpolation with sigmoid smoothing
                ratio = (value - healthy_max) / (critical_above - healthy_max)
                return self._smooth_sigmoid(1.0 - ratio)

        else:  # higher_better
            healthy_min = config.get("healthy_min", 0)
            critical_below = config.get("critical_below", healthy_min * 0.3)

            if value >= healthy_min:
                return 1.0
            elif value <= critical_below:
                return 0.0
            else:
                ratio = (value - critical_below) / (healthy_min - critical_below)
                return self._smooth_sigmoid(ratio)

    @staticmethod
    def _smooth_sigmoid(x: float) -> float:
        """
        Apply sigmoid smoothing to a 0-1 value.

        Produces S-curve that is gentle near boundaries and steep in the middle,
        creating a more natural scoring curve.

        Args:
            x: Input value between 0 and 1

        Returns:
            Sigmoid-smoothed value between 0 and 1
        """
        # Shift and scale sigmoid: steepness=6, midpoint=0.5
        return 1.0 / (1.0 + math.exp(-6 * (x - 0.5)))

    # =========================================================================
    # Trend Computation
    # =========================================================================

    def _compute_trend(
        self, metric_names: list[str], current_start: str, current_score: float = 50.0
    ) -> Optional[dict]:
        """
        Compare current period health against the previous period.

        Args:
            metric_names: List of metric names
            current_start: Start date of the current evaluation period

        Returns:
            Trend dict with direction and magnitude, or None if insufficient data
        """
        try:
            prev_end = current_start
            prev_start = (
                datetime.fromisoformat(current_start) - timedelta(days=self.lookback_days)
            ).strftime("%Y-%m-%d")

            prev_metrics = self.storage.read_gold_metrics(
                metric_names=metric_names,
                start_date=prev_start,
                end_date=prev_end,
            )

            if not prev_metrics:
                return None

            prev_map = self._build_latest_metric_map(prev_metrics)

            # Score previous period
            prev_financial = self._score_domain("financial", FINANCIAL_METRICS, prev_map)
            prev_operational = self._score_domain("operational", OPERATIONAL_METRICS, prev_map)
            prev_customer = self._score_domain("customer", CUSTOMER_METRICS, prev_map)

            prev_overall = (
                prev_financial["score"] * DOMAIN_WEIGHTS["financial"]
                + prev_operational["score"] * DOMAIN_WEIGHTS["operational"]
                + prev_customer["score"] * DOMAIN_WEIGHTS["customer"]
            )

            delta = round(current_score - prev_overall, 1)

            direction = "stable"
            if delta > 2:
                direction = "improving"
            elif delta < -2:
                direction = "degrading"

            return {
                "previous_score": round(prev_overall, 1),
                "delta": delta,
                "direction": direction,
            }

        except Exception as e:
            self.logger.debug("trend_computation_failed", error=str(e))
            return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _build_latest_metric_map(metrics: list[dict]) -> dict[str, float]:
        """
        Build a map of metric_name → latest value from raw Gold metrics.

        Args:
            metrics: List of metric dicts from storage

        Returns:
            Dict mapping metric name to most recent value
        """
        latest: dict[str, tuple[str, float]] = {}

        for m in metrics:
            name = m.get("metric_name", "")
            date = m.get("metric_date", "")
            value = m.get("metric_value")

            if not name or value is None:
                continue

            if name not in latest or date > latest[name][0]:
                latest[name] = (date, float(value))

        return {name: val for name, (_, val) in latest.items()}

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert 0-100 score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    @staticmethod
    def _score_to_status(score: float) -> str:
        """Convert 0-1 metric score to status label."""
        if score >= 0.8:
            return "healthy"
        elif score >= 0.5:
            return "degraded"
        elif score >= 0.2:
            return "warning"
        else:
            return "critical"

    @staticmethod
    def _generate_plain_explanation(
        overall_score, overall_grade, financial, operational, customer
    ) -> str:
        """Generate a plain-English explanation of the health score."""
        grade_meaning = {
            "A": "Your business is in excellent health across all areas",
            "B": "Your business is doing well overall with minor areas to watch",
            "C": "Several areas need attention — performance is below where it should be",
            "D": "Your business health is concerning — multiple areas are underperforming",
            "F": "Your business is in serious trouble — urgent action is needed",
        }

        explanation = grade_meaning.get(overall_grade, "")

        weak_areas = []
        domain_names = {"financial": "finances", "operational": "operations", "customer": "customer satisfaction"}
        for name, result in [("financial", financial), ("operational", operational), ("customer", customer)]:
            if result["score"] < 60:
                weak_areas.append(domain_names[name])

        strong_areas = []
        for name, result in [("financial", financial), ("operational", operational), ("customer", customer)]:
            if result["score"] >= 80:
                strong_areas.append(domain_names[name])

        if weak_areas:
            explanation += f". The weakest areas are {' and '.join(weak_areas)}"
        if strong_areas:
            explanation += f". {' and '.join(strong_areas).capitalize()} {'is' if len(strong_areas) == 1 else 'are'} performing well"

        # Find the worst individual metric
        worst_metric = None
        worst_score = 100
        for result in [financial, operational, customer]:
            for metric_name, metric_data in result.get("metrics", {}).items():
                score = metric_data.get("score")
                if score is not None and score < worst_score:
                    worst_score = score
                    worst_metric = metric_name

        if worst_metric and worst_score < 50:
            readable_name = worst_metric.replace("_", " ")
            explanation += f". Your biggest concern right now is {readable_name}"

        return explanation + "."

    # =========================================================================
    # SRE-Engineer Agent: Error Budget & Burn Rate
    # =========================================================================

    def compute_error_budget(
        self,
        slo_target: float = 99.5,
        window_days: int = 30,
    ) -> dict:
        """
        Compute SLO error budget for the business reliability score.

        Per sre-engineer agent: tracks SLO compliance, error budget remaining,
        and burn rate to enable data-driven feature velocity vs reliability
        trade-off decisions.

        The error budget represents the allowable degradation from the SLO
        target within the measurement window. When the budget is exhausted,
        the team should freeze feature work and focus on reliability.

        Args:
            slo_target: SLO target as percentage (default 99.5% = score >= 99.5)
                        For BRS this means overall_score >= slo_target on any given day.
            window_days: Error budget measurement window (default 30 days)

        Returns:
            Dict with:
                "slo_target": float — the target percentage
                "window_days": int — measurement window
                "total_budget_minutes": float — total allowable downtime
                "budget_consumed_minutes": float — budget used so far
                "budget_remaining_pct": float — percentage of budget remaining
                "burn_rate": float — current burn rate (1.0 = nominal)
                "budget_exhaustion_date": str | None — projected date budget runs out
                "status": str — "healthy" / "warning" / "critical" / "exhausted"
        """
        from datetime import datetime, timedelta

        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (
            datetime.utcnow() - timedelta(days=window_days)
        ).strftime("%Y-%m-%d")

        # Fetch all health-related metrics for the window
        all_metric_names = (
            list(FINANCIAL_METRICS.keys())
            + list(OPERATIONAL_METRICS.keys())
            + list(CUSTOMER_METRICS.keys())
        )

        metrics = self.storage.read_gold_metrics(
            metric_names=all_metric_names,
            start_date=start_date,
            end_date=end_date,
        )

        # Compute daily health scores
        daily_scores = self._compute_daily_scores(metrics, window_days)

        # Error budget calculation
        # Total budget = (100 - slo_target)% of total window minutes
        total_window_minutes = window_days * 24 * 60
        budget_fraction = (100.0 - slo_target) / 100.0
        total_budget_minutes = total_window_minutes * budget_fraction

        # Count minutes where score was below threshold (SLO breached)
        slo_score_threshold = slo_target  # Map SLO% to score threshold
        breached_days = sum(1 for s in daily_scores if s < slo_score_threshold)
        budget_consumed_minutes = breached_days * 24 * 60  # Full day granularity

        budget_remaining_pct = max(0.0, (
            (total_budget_minutes - budget_consumed_minutes) / total_budget_minutes * 100
        )) if total_budget_minutes > 0 else 100.0

        # Burn rate: how fast are we consuming budget relative to nominal
        # Nominal burn rate = 1.0 means we'd exactly exhaust budget at window end
        elapsed_days = len(daily_scores) if daily_scores else 1
        expected_consumption = (elapsed_days / window_days) * total_budget_minutes
        actual_consumption = budget_consumed_minutes
        burn_rate = (
            actual_consumption / expected_consumption
            if expected_consumption > 0
            else 0.0
        )

        # Project exhaustion date
        exhaustion_date = None
        if burn_rate > 1.0 and budget_remaining_pct > 0:
            remaining_minutes = total_budget_minutes - budget_consumed_minutes
            minutes_per_day = (budget_consumed_minutes / max(elapsed_days, 1))
            if minutes_per_day > 0:
                days_until_exhaustion = remaining_minutes / minutes_per_day
                exhaustion_date = (
                    datetime.utcnow() + timedelta(days=days_until_exhaustion)
                ).strftime("%Y-%m-%d")

        # Status classification
        if budget_remaining_pct <= 0:
            status = "exhausted"
        elif budget_remaining_pct <= 25 or burn_rate > 2.0:
            status = "critical"
        elif budget_remaining_pct <= 50 or burn_rate > 1.5:
            status = "warning"
        else:
            status = "healthy"

        result = {
            "slo_target": slo_target,
            "window_days": window_days,
            "total_budget_minutes": round(total_budget_minutes, 1),
            "budget_consumed_minutes": round(budget_consumed_minutes, 1),
            "budget_remaining_pct": round(budget_remaining_pct, 1),
            "burn_rate": round(burn_rate, 2),
            "budget_exhaustion_date": exhaustion_date,
            "status": status,
            "daily_scores_count": len(daily_scores),
            "breached_days": breached_days,
        }

        self.logger.info(
            "error_budget_computed",
            slo_target=slo_target,
            budget_remaining_pct=result["budget_remaining_pct"],
            burn_rate=result["burn_rate"],
            status=status,
        )

        return result

    def _compute_daily_scores(
        self, metrics: list[dict], window_days: int
    ) -> list[float]:
        """
        Compute daily health scores from raw Gold metrics.

        Groups metrics by date, scores each day, and returns a list
        of daily overall scores.

        Args:
            metrics: Raw Gold metric records
            window_days: Number of days in the window

        Returns:
            List of daily overall scores (0-100)
        """
        from collections import defaultdict

        daily_metrics: dict[str, list[dict]] = defaultdict(list)
        for m in metrics:
            date = m.get("metric_date", "")
            if date:
                daily_metrics[date].append(m)

        daily_scores = []
        for date in sorted(daily_metrics.keys()):
            day_map = self._build_latest_metric_map(daily_metrics[date])

            fin = self._score_domain("financial", FINANCIAL_METRICS, day_map)
            ops = self._score_domain("operational", OPERATIONAL_METRICS, day_map)
            cust = self._score_domain("customer", CUSTOMER_METRICS, day_map)

            overall = (
                fin["score"] * DOMAIN_WEIGHTS["financial"]
                + ops["score"] * DOMAIN_WEIGHTS["operational"]
                + cust["score"] * DOMAIN_WEIGHTS["customer"]
            )
            daily_scores.append(round(overall, 1))

        return daily_scores
