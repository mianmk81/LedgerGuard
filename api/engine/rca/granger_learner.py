"""
Granger Causality Learner — Learned edge strengths for RCA graph.

Loads pre-trained Granger causality artifacts from models/causal_graph/granger_edges.json.
Exposes per-edge statistical strengths (0.1–1.0) learned from Gold-layer time series data.

Used by BusinessDependencyGraph to augment compute_graph_proximity() with data-driven
edge weights instead of treating all edges equally.

Training: python scripts/train_causal_graph.py
Artifact: models/causal_graph/granger_edges.json
"""

import json
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

# Module-level singleton cache (same pattern as churn_predictor.py)
_GRANGER_CACHE: Optional["GrangerLearner"] = None


class GrangerLearner:
    """
    Loads and serves Granger causality edge strengths.

    Edge strengths are statistical confidence scores (0.1–1.0) derived from
    Granger causality tests on Gold-layer daily metric time series. A strength
    of 1.0 means the source metric strongly predicts the target; 0.1 is the
    floor for non-significant edges (neutral, doesn't penalize).

    Attributes:
        is_loaded: True if artifact was found and loaded successfully
    """

    def __init__(self, models_dir: Path):
        self._edges: dict[str, dict] = {}
        self._discovered: dict[str, dict] = {}
        self._metadata: dict = {}
        self._loaded = False
        self._load(models_dir)

    def _load(self, models_dir: Path) -> None:
        path = models_dir / "causal_graph" / "granger_edges.json"
        if not path.exists():
            logger.info(
                "granger_artifact_not_found",
                path=str(path),
                fallback="static_equal_weights",
                hint="Run: python scripts/train_causal_graph.py",
            )
            return

        try:
            with open(path) as f:
                data = json.load(f)

            self._edges = data.get("edges", {})
            self._discovered = data.get("discovered_edges", {})
            self._metadata = {
                "version": data.get("version"),
                "trained_at": data.get("trained_at"),
                "n_days": data.get("n_days"),
                "p_threshold": data.get("p_threshold"),
                "max_lag": data.get("max_lag"),
                "summary": data.get("summary", {}),
            }
            self._loaded = True

            logger.info(
                "granger_learner_loaded",
                artifact_path=str(path),
                n_edges=len(self._edges),
                n_discovered=len(self._discovered),
                trained_at=self._metadata.get("trained_at"),
                n_significant=self._metadata.get("summary", {}).get("edges_significant", "?"),
            )

        except Exception as e:
            logger.error(
                "granger_artifact_load_failed",
                path=str(path),
                error=str(e),
                fallback="static_equal_weights",
            )

    def get_edge_strength(self, source: str, target: str) -> float:
        """
        Return the learned Granger causality strength for an edge.

        Args:
            source: Upstream metric name
            target: Downstream metric name

        Returns:
            Strength in [0.1, 1.0]. Returns 1.0 (neutral) when:
            - Artifact not loaded
            - Edge not found in artifact
            This ensures zero regression when training hasn't been run.
        """
        if not self._loaded:
            return 1.0

        key = f"{source}\u2192{target}"  # "→" separator matching training output
        edge_data = self._edges.get(key) or self._discovered.get(key)
        if edge_data is None:
            return 1.0  # Unknown edge — neutral weight

        return float(edge_data.get("strength", 1.0))

    def get_edge_metadata(self, source: str, target: str) -> dict:
        """
        Return full Granger test metadata for an edge.

        Returns:
            Dict with strength, p_value, f_stat, optimal_lag, significant.
            Empty dict if edge not found.
        """
        key = f"{source}\u2192{target}"
        return dict(self._edges.get(key) or self._discovered.get(key) or {})

    def get_all_edges(self) -> dict[str, dict]:
        """Return all learned edges (hardcoded + discovered)."""
        return {**self._edges, **self._discovered}

    @property
    def is_loaded(self) -> bool:
        """True if Granger artifact was successfully loaded."""
        return self._loaded

    @property
    def metadata(self) -> dict:
        """Training metadata (version, trained_at, summary stats)."""
        return dict(self._metadata)

    @property
    def n_edges(self) -> int:
        """Number of edges with learned strengths."""
        return len(self._edges) + len(self._discovered)


def get_granger_learner() -> GrangerLearner:
    """
    Singleton accessor for GrangerLearner.

    Returns the cached instance, creating it on first call using the
    models_dir from application settings. Safe to call multiple times.

    Returns:
        GrangerLearner instance (may have is_loaded=False if artifact missing)
    """
    global _GRANGER_CACHE
    if _GRANGER_CACHE is None:
        try:
            from api.config import get_settings
            models_dir = Path(get_settings().models_dir)
        except Exception:
            models_dir = Path("./models")
        _GRANGER_CACHE = GrangerLearner(models_dir)
    return _GRANGER_CACHE


def reset_granger_cache() -> None:
    """Reset singleton cache. Used in tests to reload with fresh paths."""
    global _GRANGER_CACHE
    _GRANGER_CACHE = None
