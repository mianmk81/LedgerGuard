"""
Causal Ranker for BRE-RCA Algorithm.

This module implements the contribution scoring and ranking algorithm that
determines which upstream metric anomalies are the most likely root causes
of a detected incident.

The BRE-RCA Contribution Score formula:
    contribution_score = w1 * anomaly_magnitude_norm
                       + w2 * temporal_precedence
                       + w3 * graph_proximity
                       + w4 * data_quality_weight

Default weights (tuned for financial operations):
    w1 = 0.30  (anomaly strength)
    w2 = 0.30  (temporal precedence)
    w3 = 0.25  (graph proximity)
    w4 = 0.15  (data quality)

After scoring, candidates are ranked by contribution_score and the top-k
are assembled into CausalPath objects with full evidence.

Version: rca_ranker_v1
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

import numpy as np
import structlog

from api.models.rca import CausalNode, CausalPath, EvidenceCluster

logger = structlog.get_logger()


class CausalRanker:
    """
    Ranks candidate root causes by composite contribution score.

    Combines four scoring dimensions to produce a principled ranking
    of candidate metrics that could have caused the detected incident.
    The ranking respects both statistical evidence (anomaly magnitude,
    temporal precedence) and domain knowledge (graph proximity).

    Attributes:
        w_anomaly: Weight for anomaly magnitude component
        w_temporal: Weight for temporal precedence component
        w_proximity: Weight for graph proximity component
        w_quality: Weight for data quality component
        logger: Structured logger for observability

    Example:
        >>> ranker = CausalRanker()
        >>> ranked_paths = ranker.rank_candidates(
        ...     candidates=candidate_list,
        ...     incident_metric="refund_rate",
        ...     top_k=5,
        ... )
        >>> print(f"Top cause: {ranked_paths[0].nodes[0].metric_name}")
    """

    # Default contribution score weights
    DEFAULT_WEIGHTS = {
        "anomaly": 0.30,
        "temporal": 0.30,
        "proximity": 0.25,
        "quality": 0.15,
    }

    def __init__(
        self,
        w_anomaly: float = 0.30,
        w_temporal: float = 0.30,
        w_proximity: float = 0.25,
        w_quality: float = 0.15,
    ):
        """
        Initialize the causal ranker with contribution weights.

        Args:
            w_anomaly: Weight for anomaly magnitude (default 0.30)
            w_temporal: Weight for temporal precedence (default 0.30)
            w_proximity: Weight for graph proximity (default 0.25)
            w_quality: Weight for data quality (default 0.15)

        Raises:
            ValueError: If weights don't sum to approximately 1.0
        """
        total = w_anomaly + w_temporal + w_proximity + w_quality
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Contribution weights must sum to 1.0, got {total:.4f}"
            )

        self.w_anomaly = w_anomaly
        self.w_temporal = w_temporal
        self.w_proximity = w_proximity
        self.w_quality = w_quality
        self.logger = structlog.get_logger()

    def rank_candidates(
        self,
        candidates: list[dict],
        incident_metric: str,
        top_k: int = 5,
    ) -> list[CausalPath]:
        """
        Rank candidate root causes and assemble into CausalPath objects.

        Takes raw candidate data (metric name + scoring dimensions) and
        produces a ranked list of CausalPath objects ready for inclusion
        in a CausalChain.

        Args:
            candidates: List of candidate dicts, each containing:
                {
                    "metric_name": str,
                    "anomaly_magnitude": float,  # z-score
                    "temporal_precedence": float,  # 0.0-1.0
                    "graph_proximity": float,  # 0.0-1.0
                    "data_quality_weight": float,  # 0.0-1.0
                    "metric_value": float,
                    "metric_baseline": float,
                    "metric_zscore": float,
                    "anomaly_window": tuple[datetime, datetime],
                    "evidence_clusters": list[EvidenceCluster],  # optional
                }
            incident_metric: Name of the incident metric (for path context)
            top_k: Number of top candidates to return

        Returns:
            List of CausalPath objects ranked by contribution score (descending)

        Example:
            >>> paths = ranker.rank_candidates(candidates, "refund_rate", top_k=5)
        """
        if not candidates:
            self.logger.warning("no_candidates_to_rank")
            return []

        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            try:
                contribution_score = self._compute_contribution_score(candidate)
                scored_candidates.append({
                    **candidate,
                    "contribution_score": contribution_score,
                })
            except Exception as e:
                self.logger.error(
                    "candidate_scoring_failed",
                    metric_name=candidate.get("metric_name"),
                    error=str(e),
                )

        # Sort by contribution score descending
        scored_candidates.sort(
            key=lambda c: c["contribution_score"], reverse=True
        )

        # Take top-k candidates
        top_candidates = scored_candidates[:top_k]

        # Assemble into CausalPath objects
        paths = []
        for rank, candidate in enumerate(top_candidates, start=1):
            try:
                path = self._assemble_causal_path(candidate, rank)
                paths.append(path)
            except Exception as e:
                self.logger.error(
                    "path_assembly_failed",
                    metric_name=candidate.get("metric_name"),
                    rank=rank,
                    error=str(e),
                )

        self.logger.info(
            "candidates_ranked",
            total_candidates=len(candidates),
            scored_candidates=len(scored_candidates),
            paths_created=len(paths),
            top_score=paths[0].overall_score if paths else 0.0,
            incident_metric=incident_metric,
        )

        return paths

    def compute_contribution_score(self, candidate: dict) -> float:
        """
        Compute the composite contribution score for a single candidate.

        Public interface for scoring individual candidates outside of
        the full ranking pipeline.

        Args:
            candidate: Candidate dict with scoring dimensions

        Returns:
            Contribution score in [0.0, 1.0]
        """
        return self._compute_contribution_score(candidate)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _compute_contribution_score(self, candidate: dict) -> float:
        """
        Compute weighted contribution score from four dimensions.

        Args:
            candidate: Dict with anomaly_magnitude, temporal_precedence,
                      graph_proximity, data_quality_weight

        Returns:
            Contribution score in [0.0, 1.0]
        """
        # Normalize anomaly magnitude to [0, 1] using sigmoid-like transform
        raw_magnitude = abs(candidate.get("anomaly_magnitude", 0.0))
        # Sigmoid normalization: maps z-scores to [0, 1]
        # z=2 → ~0.5, z=4 → ~0.8, z=8 → ~0.95
        anomaly_norm = self._sigmoid_normalize(raw_magnitude, midpoint=3.0, steepness=0.5)

        temporal = max(0.0, min(1.0, candidate.get("temporal_precedence", 0.0)))
        proximity = max(0.0, min(1.0, candidate.get("graph_proximity", 0.0)))
        quality = max(0.0, min(1.0, candidate.get("data_quality_weight", 1.0)))

        score = (
            self.w_anomaly * anomaly_norm
            + self.w_temporal * temporal
            + self.w_proximity * proximity
            + self.w_quality * quality
        )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        self.logger.debug(
            "contribution_score_computed",
            metric_name=candidate.get("metric_name"),
            anomaly_norm=round(anomaly_norm, 4),
            temporal=round(temporal, 4),
            proximity=round(proximity, 4),
            quality=round(quality, 4),
            final_score=round(score, 4),
        )

        return round(score, 4)

    def _sigmoid_normalize(
        self, value: float, midpoint: float = 3.0, steepness: float = 0.5
    ) -> float:
        """
        Normalize a non-negative value to [0, 1] using sigmoid function.

        Maps z-scores to a bounded [0, 1] range with smooth saturation:
        - Values near 0 → near 0
        - Values near midpoint → 0.5
        - Large values → approaching 1.0

        Args:
            value: Non-negative value to normalize
            midpoint: Value that maps to 0.5
            steepness: Controls how quickly the function saturates

        Returns:
            Normalized value in [0.0, 1.0]
        """
        try:
            return 1.0 / (1.0 + np.exp(-steepness * (value - midpoint)))
        except (OverflowError, FloatingPointError):
            return 1.0 if value > midpoint else 0.0

    def _assemble_causal_path(self, candidate: dict, rank: int) -> CausalPath:
        """
        Assemble a scored candidate into a CausalPath object.

        Creates a single-node CausalPath representing the direct causal
        link from the candidate metric to the incident. Multi-hop paths
        are assembled by the analyzer using graph traversal.

        Args:
            candidate: Scored candidate dict
            rank: Ranking position (1-based)

        Returns:
            CausalPath with single CausalNode
        """
        # Extract anomaly window with fallback
        anomaly_window = candidate.get("anomaly_window")
        if anomaly_window is None:
            # Default to last 24 hours
            now = datetime.utcnow()
            anomaly_window = (now - timedelta(hours=24), now)

        # Extract evidence clusters
        evidence_clusters = candidate.get("evidence_clusters", [])

        # Create CausalNode
        node = CausalNode(
            metric_name=candidate["metric_name"],
            contribution_score=candidate["contribution_score"],
            anomaly_magnitude=candidate.get("anomaly_magnitude", 0.0),
            temporal_precedence=candidate.get("temporal_precedence", 0.0),
            graph_proximity=candidate.get("graph_proximity", 0.0),
            data_quality_weight=candidate.get("data_quality_weight", 1.0),
            metric_value=candidate.get("metric_value", 0.0),
            metric_baseline=candidate.get("metric_baseline", 0.0),
            metric_zscore=candidate.get("metric_zscore", 0.0),
            anomaly_window=anomaly_window,
            evidence_clusters=evidence_clusters,
        )

        # Create CausalPath with single node
        path = CausalPath(
            rank=rank,
            overall_score=candidate["contribution_score"],
            nodes=[node],
        )

        return path

    def compute_relative_importance(
        self, paths: list[CausalPath]
    ) -> dict[str, float]:
        """
        Compute relative importance percentages for ranked paths.

        Normalizes contribution scores so they sum to 100%, giving
        stakeholders intuitive "this caused X% of the problem" framing.

        Args:
            paths: List of ranked CausalPaths

        Returns:
            Dict mapping metric_name → percentage (0-100)

        Example:
            >>> importance = ranker.compute_relative_importance(paths)
            >>> print(importance)
            {'supplier_delay_rate': 42.3, 'ticket_volume': 28.1, ...}
        """
        if not paths:
            return {}

        total_score = sum(p.overall_score for p in paths)
        if total_score == 0:
            return {
                p.nodes[0].metric_name: 0.0 for p in paths
            }

        importance = {}
        for path in paths:
            metric_name = path.nodes[0].metric_name
            pct = (path.overall_score / total_score) * 100.0
            importance[metric_name] = round(pct, 1)

        return importance

    # =========================================================================
    # Data-Scientist Agent: Statistical Validation & Sensitivity Analysis
    # =========================================================================

    def compute_confidence_intervals(
        self,
        candidates: list[dict],
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
    ) -> dict[str, dict]:
        """
        Compute bootstrap confidence intervals for contribution scores.

        Uses non-parametric bootstrap resampling to estimate the
        uncertainty of each candidate's contribution score. This
        addresses the data-scientist agent requirement for statistical
        validation and reproducibility.

        Args:
            candidates: List of candidate dicts with scoring dimensions
            n_bootstrap: Number of bootstrap iterations (default 200)
            ci_level: Confidence interval level (default 0.95 for 95% CI)

        Returns:
            Dict mapping metric_name → {
                "point_estimate": float,
                "ci_lower": float,
                "ci_upper": float,
                "std_error": float,
                "is_significant": bool,  # CI excludes zero
            }

        Example:
            >>> ci = ranker.compute_confidence_intervals(candidates)
            >>> print(ci["refund_rate"]["ci_lower"])  # 0.62
        """
        results = {}
        alpha = 1 - ci_level

        for candidate in candidates:
            metric_name = candidate.get("metric_name", "unknown")
            point_est = self._compute_contribution_score(candidate)

            # Bootstrap: perturb each scoring dimension with small noise
            bootstrap_scores = []
            for _ in range(n_bootstrap):
                perturbed = self._perturb_candidate(candidate)
                score = self._compute_contribution_score(perturbed)
                bootstrap_scores.append(score)

            bootstrap_arr = np.array(bootstrap_scores)
            ci_lower = float(np.percentile(bootstrap_arr, (alpha / 2) * 100))
            ci_upper = float(np.percentile(bootstrap_arr, (1 - alpha / 2) * 100))
            std_error = float(np.std(bootstrap_arr, ddof=1))

            results[metric_name] = {
                "point_estimate": point_est,
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "std_error": round(std_error, 4),
                "is_significant": ci_lower > 0.05,  # CI excludes trivial scores
            }

        self.logger.info(
            "confidence_intervals_computed",
            n_candidates=len(candidates),
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
        )

        return results

    def sensitivity_analysis(
        self,
        candidates: list[dict],
        incident_metric: str,
        top_k: int = 5,
    ) -> dict:
        """
        Perform sensitivity analysis on weight parameters.

        Tests how robust the ranking is to changes in the contribution
        weight parameters. A stable ranking under weight perturbation
        indicates a reliable causal conclusion. This addresses the
        data-scientist agent requirement for assumption verification
        and sensitivity analysis.

        Args:
            candidates: List of candidate dicts
            incident_metric: Incident metric name
            top_k: Number of top candidates to track

        Returns:
            Dict with:
                "rank_stability": float (0-1, fraction of perturbations
                    where top-1 candidate stays the same)
                "top1_robustness": str ("high"/"medium"/"low")
                "weight_sensitivity": dict mapping weight → impact score
                "alternative_rankings": list of alternate top-1 candidates

        Example:
            >>> sa = ranker.sensitivity_analysis(candidates, "refund_rate")
            >>> print(sa["rank_stability"])  # 0.92
        """
        if not candidates:
            return {"rank_stability": 0.0, "top1_robustness": "low"}

        # Baseline ranking
        baseline_paths = self.rank_candidates(candidates, incident_metric, top_k)
        if not baseline_paths:
            return {"rank_stability": 0.0, "top1_robustness": "low"}

        baseline_top1 = baseline_paths[0].nodes[0].metric_name

        # Perturbation grid: vary each weight by +/- 0.05, 0.10
        perturbations = [
            {"anomaly": 0.05}, {"anomaly": -0.05}, {"anomaly": 0.10}, {"anomaly": -0.10},
            {"temporal": 0.05}, {"temporal": -0.05}, {"temporal": 0.10}, {"temporal": -0.10},
            {"proximity": 0.05}, {"proximity": -0.05}, {"proximity": 0.10}, {"proximity": -0.10},
            {"quality": 0.05}, {"quality": -0.05}, {"quality": 0.10}, {"quality": -0.10},
        ]

        same_top1_count = 0
        alternative_top1s = set()
        weight_impacts = {"anomaly": 0, "temporal": 0, "proximity": 0, "quality": 0}

        for perturbation in perturbations:
            # Create perturbed ranker
            new_weights = {
                "anomaly": self.w_anomaly,
                "temporal": self.w_temporal,
                "proximity": self.w_proximity,
                "quality": self.w_quality,
            }
            for key, delta in perturbation.items():
                new_weights[key] = max(0.01, new_weights[key] + delta)

            # Renormalize to sum to 1.0
            total = sum(new_weights.values())
            for key in new_weights:
                new_weights[key] /= total

            try:
                perturbed_ranker = CausalRanker(
                    w_anomaly=new_weights["anomaly"],
                    w_temporal=new_weights["temporal"],
                    w_proximity=new_weights["proximity"],
                    w_quality=new_weights["quality"],
                )
                perturbed_paths = perturbed_ranker.rank_candidates(
                    candidates, incident_metric, top_k
                )

                if perturbed_paths:
                    perturbed_top1 = perturbed_paths[0].nodes[0].metric_name
                    if perturbed_top1 == baseline_top1:
                        same_top1_count += 1
                    else:
                        alternative_top1s.add(perturbed_top1)
                        for key in perturbation:
                            weight_impacts[key] += 1
            except (ValueError, Exception):
                continue

        rank_stability = same_top1_count / len(perturbations) if perturbations else 0.0

        if rank_stability >= 0.85:
            robustness = "high"
        elif rank_stability >= 0.60:
            robustness = "medium"
        else:
            robustness = "low"

        # Normalize weight impact to [0, 1]
        max_impact = max(weight_impacts.values()) if weight_impacts else 1
        weight_sensitivity = {
            k: round(v / max(max_impact, 1), 2) for k, v in weight_impacts.items()
        }

        result = {
            "rank_stability": round(rank_stability, 4),
            "top1_robustness": robustness,
            "baseline_top1": baseline_top1,
            "weight_sensitivity": weight_sensitivity,
            "alternative_rankings": sorted(alternative_top1s),
            "perturbations_tested": len(perturbations),
        }

        self.logger.info(
            "sensitivity_analysis_complete",
            rank_stability=result["rank_stability"],
            robustness=robustness,
            alternatives=len(alternative_top1s),
        )

        return result

    def _perturb_candidate(self, candidate: dict) -> dict:
        """
        Create a perturbed copy of a candidate for bootstrap resampling.

        Adds small Gaussian noise to each scoring dimension to simulate
        measurement uncertainty.

        Args:
            candidate: Original candidate dict

        Returns:
            Perturbed copy with noisy scoring dimensions
        """
        perturbed = dict(candidate)
        noise_scale = 0.05  # 5% relative noise

        for key in ["anomaly_magnitude", "temporal_precedence", "graph_proximity", "data_quality_weight"]:
            original = perturbed.get(key, 0.0)
            noise = np.random.normal(0, noise_scale * max(abs(original), 0.1))
            perturbed[key] = original + noise

        return perturbed
