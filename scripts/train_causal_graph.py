"""
Train Granger Causal Graph — Learn edge strengths from Gold-layer time series.

Reads 6 months of daily Gold metrics from DuckDB, runs pairwise Granger causality
tests on all 21 BUSINESS_DEPENDENCY_EDGES, and saves learned edge strengths to
models/causal_graph/granger_edges.json.

Optionally discovers new causal edges not in the hardcoded graph (--discover flag).

Usage:
    python scripts/train_causal_graph.py
    python scripts/train_causal_graph.py --max-lag 7 --p-threshold 0.05 --discover
    python scripts/train_causal_graph.py --db-path ./data/ledgerguard.duckdb

Output:
    models/causal_graph/granger_edges.json   — learned edge strengths
    MLflow experiment: ledgerguard-causal-graph
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = structlog.get_logger()

# All 21 hardcoded causal edges to test
BUSINESS_DEPENDENCY_EDGES = [
    ("supplier_delay_rate", "delivery_delay_rate"),
    ("supplier_delay_rate", "fulfillment_backlog"),
    ("order_volume", "fulfillment_backlog"),
    ("fulfillment_backlog", "delivery_delay_rate"),
    ("delivery_delay_rate", "ticket_volume"),
    ("delivery_delay_rate", "review_score_avg"),
    ("ticket_volume", "ticket_backlog"),
    ("ticket_backlog", "avg_resolution_time"),
    ("avg_resolution_time", "review_score_avg"),
    ("review_score_avg", "churn_proxy"),
    ("ticket_volume", "churn_proxy"),
    ("churn_proxy", "daily_revenue"),
    ("refund_rate", "margin_proxy"),
    ("refund_rate", "daily_revenue"),
    ("daily_revenue", "margin_proxy"),
    ("daily_expenses", "expense_ratio"),
    ("expense_ratio", "margin_proxy"),
    ("margin_proxy", "net_cash_proxy"),
    ("dso_proxy", "ar_aging_amount"),
    ("ar_aging_amount", "net_cash_proxy"),
]

# All 27 Gold metrics (for discovery pass)
ALL_METRICS = [
    "daily_revenue", "daily_expenses", "daily_refunds", "refund_rate",
    "net_cash_proxy", "expense_ratio", "margin_proxy", "dso_proxy",
    "ar_aging_amount", "ar_overdue_count", "dpo_proxy",
    "order_volume", "delivery_count", "late_delivery_count", "delivery_delay_rate",
    "fulfillment_backlog", "avg_delivery_delay_days", "supplier_delay_rate",
    "supplier_delay_severity",
    "ticket_volume", "ticket_close_volume", "ticket_backlog", "avg_resolution_time",
    "review_score_avg", "review_score_trend", "churn_proxy", "customer_concentration",
]

MIN_SAMPLES = 20  # Minimum data points required to run Granger test


def run_granger_test(df: pd.DataFrame, source: str, target: str, max_lag: int) -> dict:
    """
    Run Granger causality test: does source Granger-cause target?

    Returns dict with p_value, f_stat, optimal_lag, strength, significant.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        logger.error("statsmodels_not_installed", hint="pip install statsmodels>=0.14.0")
        raise

    # Need both columns with enough data
    cols = [target, source]
    sub = df[cols].dropna()

    if len(sub) < MIN_SAMPLES + max_lag:
        return {
            "strength": 1.0,
            "p_value": None,
            "f_stat": None,
            "optimal_lag": None,
            "significant": False,
            "skipped": True,
            "reason": f"insufficient data: {len(sub)} rows < {MIN_SAMPLES + max_lag}",
        }

    try:
        results = grangercausalitytests(sub, maxlag=max_lag, verbose=False)

        # Find optimal lag (lowest p-value across tested lags)
        best_lag = min(results.keys(), key=lambda lag: results[lag][0]["ssr_ftest"][1])
        p_value = float(results[best_lag][0]["ssr_ftest"][1])
        f_stat = float(results[best_lag][0]["ssr_ftest"][0])

        return {
            "p_value": round(p_value, 6),
            "f_stat": round(f_stat, 4),
            "optimal_lag": best_lag,
            "significant": bool(p_value < 0.05),  # Explicit bool for JSON serialization
            "skipped": False,
        }

    except Exception as e:
        logger.warning("granger_test_failed", source=source, target=target, error=str(e))
        return {
            "strength": 1.0,
            "p_value": None,
            "f_stat": None,
            "optimal_lag": None,
            "significant": False,
            "skipped": True,
            "reason": str(e),
        }


def compute_strength(p_value: float | None, p_threshold: float, significant: bool) -> float:
    """
    Convert p-value to edge strength in [0.1, 1.0].

    Significant edges (p < threshold): strength = 1 - p_value (capped at 0.99)
    Non-significant edges: floor at 0.1 (neutral — doesn't penalize)
    Missing p-value: 1.0 (no opinion — neutral weight)
    """
    if p_value is None:
        return 1.0
    if significant and p_value < p_threshold:
        return round(max(0.1, min(0.99, 1.0 - p_value)), 4)
    return 0.1  # Non-significant — weak causal evidence


def has_cycle(edges: list[tuple[str, str]], new_edge: tuple[str, str]) -> bool:
    """Check if adding new_edge to edges would create a cycle."""
    import networkx as nx
    g = nx.DiGraph()
    g.add_edges_from(edges)
    g.add_edge(*new_edge)
    return not nx.is_directed_acyclic_graph(g)


def load_gold_metrics(storage, start_date: str | None = None) -> pd.DataFrame:
    """
    Load all Gold-layer metrics from DuckDB and pivot to wide DataFrame.

    Returns DataFrame with DatetimeIndex and metric names as columns.
    """
    logger.info("loading_gold_metrics", start_date=start_date or "all")
    records = storage.read_gold_metrics(
        metric_names=ALL_METRICS,
        start_date=start_date,
    )

    if not records:
        logger.error("no_gold_metrics_found", hint="Run: python scripts/seed_sandbox.py --mode local")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["metric_date"] = pd.to_datetime(df["metric_date"])
    df = df.pivot_table(index="metric_date", columns="metric_name", values="metric_value", aggfunc="mean")
    df = df.sort_index()

    logger.info(
        "gold_metrics_loaded",
        n_days=len(df),
        n_metrics=len(df.columns),
        date_range=f"{df.index.min().date()} to {df.index.max().date()}",
    )
    return df


def train(
    db_path: str | None = None,
    max_lag: int = 7,
    p_threshold: float = 0.05,
    discover: bool = False,
    output_dir: Path | None = None,
    use_mlflow: bool = True,
) -> dict:
    """
    Full training pipeline.

    Returns the artifact dict (same as written to JSON).
    """
    # ── Setup ──────────────────────────────────────────────────────────────────
    from api.config import get_settings
    from api.storage.duckdb_storage import DuckDBStorage

    settings = get_settings()
    resolved_db = db_path or settings.db_path
    models_dir = output_dir or Path(settings.models_dir)
    out_dir = models_dir / "causal_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "causal_graph_training_started",
        db_path=resolved_db,
        max_lag=max_lag,
        p_threshold=p_threshold,
        discover=discover,
    )

    # ── Storage ────────────────────────────────────────────────────────────────
    storage = DuckDBStorage(db_path=resolved_db)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = load_gold_metrics(storage)
    if df.empty:
        logger.error("training_aborted", reason="no gold metrics data")
        return {}

    n_days = len(df)

    # ── Test hardcoded edges ───────────────────────────────────────────────────
    edges_result: dict[str, dict] = {}
    n_significant = 0
    n_skipped = 0

    print(f"\n{'='*60}")
    print(f"Granger Causality Tests - {len(BUSINESS_DEPENDENCY_EDGES)} edges, max_lag={max_lag}")
    print(f"Data: {n_days} days, p_threshold={p_threshold}")
    print(f"{'='*60}")
    print(f"{'Edge':<50} {'p-val':>8} {'F-stat':>8} {'lag':>4} {'strength':>9} {'sig':>5}")
    print(f"{'-'*60}")

    for source, target in BUSINESS_DEPENDENCY_EDGES:
        if source not in df.columns or target not in df.columns:
            logger.warning("metric_missing_from_gold", source=source, target=target)
            continue

        result = run_granger_test(df, source, target, max_lag)
        strength = compute_strength(result.get("p_value"), p_threshold, result.get("significant", False))
        result["strength"] = strength

        key = f"{source}\u2192{target}"
        edges_result[key] = result

        if result.get("significant"):
            n_significant += 1
        if result.get("skipped"):
            n_skipped += 1

        # Print row
        p_str = f"{result['p_value']:.4f}" if result.get("p_value") is not None else "  skip"
        f_str = f"{result['f_stat']:.2f}" if result.get("f_stat") is not None else "     -"
        lag_str = str(result.get("optimal_lag") or "-")
        sig_str = "Y" if result.get("significant") else "N"
        print(f"{source} -> {target:<40} {p_str:>8} {f_str:>8} {lag_str:>4} {strength:>9.3f} {sig_str:>5}")

    print(f"{'='*60}")
    print(f"Significant: {n_significant}/{len(BUSINESS_DEPENDENCY_EDGES) - n_skipped} tested  |  Skipped: {n_skipped}")

    # ── Discovery pass (optional) ──────────────────────────────────────────────
    discovered_edges: dict[str, dict] = {}
    n_discovered = 0

    if discover:
        existing_pairs = set(BUSINESS_DEPENDENCY_EDGES)
        current_edge_list = list(BUSINESS_DEPENDENCY_EDGES)
        print(f"\n{'='*60}")
        print(f"Discovery Pass — testing {len(ALL_METRICS)**2 - len(ALL_METRICS)} cross-pairs")
        print(f"{'='*60}")

        for source in ALL_METRICS:
            for target in ALL_METRICS:
                if source == target:
                    continue
                if (source, target) in existing_pairs:
                    continue
                if source not in df.columns or target not in df.columns:
                    continue

                result = run_granger_test(df, source, target, max_lag)
                if result.get("significant") and not result.get("skipped"):
                    # Cycle check before accepting
                    if has_cycle(current_edge_list, (source, target)):
                        logger.debug("discovered_edge_skipped_cycle", source=source, target=target)
                        continue

                    strength = compute_strength(result["p_value"], p_threshold, True)
                    result["strength"] = strength
                    key = f"{source}\u2192{target}"
                    discovered_edges[key] = result
                    current_edge_list.append((source, target))
                    n_discovered += 1
                    print(f"  DISCOVERED: {source} -> {target}  (p={result['p_value']:.4f}, strength={strength:.3f})")

        print(f"\nDiscovered {n_discovered} new significant causal edges")

    # ── Compute summary stats ──────────────────────────────────────────────────
    all_strengths = [v["strength"] for v in edges_result.values() if not v.get("skipped")]
    mean_strength = round(float(np.mean(all_strengths)), 4) if all_strengths else 0.0

    summary = {
        "edges_tested": len(BUSINESS_DEPENDENCY_EDGES) - n_skipped,
        "edges_skipped": n_skipped,
        "edges_significant": n_significant,
        "mean_strength": mean_strength,
        "new_edges_discovered": n_discovered,
    }

    # ── Build artifact ─────────────────────────────────────────────────────────
    artifact = {
        "version": "granger_v1",
        "trained_at": datetime.utcnow().isoformat(),
        "n_days": n_days,
        "p_threshold": p_threshold,
        "max_lag": max_lag,
        "edges": edges_result,
        "discovered_edges": discovered_edges,
        "summary": summary,
    }

    # ── Save artifact ──────────────────────────────────────────────────────────
    out_path = out_dir / "granger_edges.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    logger.info(
        "granger_artifact_saved",
        path=str(out_path),
        n_edges=len(edges_result),
        n_discovered=n_discovered,
        mean_strength=mean_strength,
    )
    print(f"\n[OK] Artifact saved -> {out_path}")

    # ── MLflow logging ─────────────────────────────────────────────────────────
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("ledgerguard-causal-graph")
            with mlflow.start_run(run_name="granger_causality"):
                mlflow.log_param("max_lag", max_lag)
                mlflow.log_param("p_threshold", p_threshold)
                mlflow.log_param("n_metrics", len(df.columns))
                mlflow.log_param("n_days", n_days)
                mlflow.log_param("discover", discover)

                mlflow.log_metric("n_edges_tested", summary["edges_tested"])
                mlflow.log_metric("n_significant", n_significant)
                mlflow.log_metric("mean_strength", mean_strength)
                mlflow.log_metric("n_discovered", n_discovered)
                mlflow.log_metric("significance_rate", n_significant / max(summary["edges_tested"], 1))

                mlflow.log_artifact(str(out_path))

            print("[OK] Logged to MLflow experiment: ledgerguard-causal-graph")
        except Exception as e:
            logger.warning("mlflow_logging_failed", error=str(e))

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"  Edges tested:      {summary['edges_tested']}")
    print(f"  Significant:       {summary['edges_significant']} ({100*summary['edges_significant']//max(summary['edges_tested'],1)}%)")
    print(f"  Mean strength:     {mean_strength:.3f}")
    print(f"  Discovered edges:  {n_discovered}")
    print(f"{'='*60}\n")

    return artifact


def main():
    parser = argparse.ArgumentParser(
        description="Train Granger causal graph from Gold-layer time series"
    )
    parser.add_argument("--db-path", default=None, help="DuckDB file path (default: from settings)")
    parser.add_argument("--max-lag", type=int, default=7, help="Maximum lag days for Granger test (default: 7)")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="P-value significance threshold (default: 0.05)")
    parser.add_argument("--discover", action="store_true", help="Discover new edges beyond the 21 hardcoded ones")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    train(
        db_path=args.db_path,
        max_lag=args.max_lag,
        p_threshold=args.p_threshold,
        discover=args.discover,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
