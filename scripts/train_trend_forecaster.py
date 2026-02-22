#!/usr/bin/env python3
"""
LedgerGuard — Trend Forecaster Training

Trains LightGBM models to project Gold metrics N days ahead. Replaces linear
regression in TrendDetector for improved trend projection when models exist.

Uses temporal train/test split (last 20% held out) to report test MAE for
reliable performance estimates. Avoids overfitting bias from in-sample metrics.

Uses Gold metrics from storage (run seed_sandbox.py first to populate).

Usage:
    python scripts/train_trend_forecaster.py
    python scripts/train_trend_forecaster.py --lookback 14 --projection 5 --min-days 60
    python scripts/train_trend_forecaster.py --test-fraction 0.2

Requires: Gold metrics in storage (python scripts/seed_sandbox.py --mode local)
"""

import argparse
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import get_settings
from api.engine.prediction.trend_detector import TREND_THRESHOLDS
from api.storage.duckdb_storage import DuckDBStorage

logger = structlog.get_logger()

# Lag and rolling window for features
DEFAULT_LOOKBACK = 14
DEFAULT_PROJECTION = 5
DEFAULT_MIN_TRAINING_DAYS = 60
DEFAULT_TEST_FRACTION = 0.2  # Last 20% of samples held out for test (temporal split)


def build_lag_features(values: list[float], n_lags: int = 14) -> list[float]:
    """Build feature vector from recent values and rolling stats."""
    if len(values) < n_lags:
        return []
    recent = values[-n_lags:]
    rolling_7 = values[-7:] if len(values) >= 7 else recent
    rolling_14 = values[-14:] if len(values) >= 14 else recent
    return [
        *recent,
        float(np.mean(rolling_7)),
        float(np.std(rolling_7)) if len(rolling_7) > 1 else 0.0,
        float(np.mean(rolling_14)),
        float(np.std(rolling_14)) if len(rolling_14) > 1 else 0.0,
        values[-1] - values[-2] if len(values) >= 2 else 0.0,  # 1-day trend
    ]


def train_forecaster(
    storage: DuckDBStorage,
    metric_name: str,
    lookback_days: int = DEFAULT_LOOKBACK,
    projection_days: int = DEFAULT_PROJECTION,
    min_days: int = DEFAULT_MIN_TRAINING_DAYS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    models_dir: Path | None = None,
) -> dict | None:
    """
    Train LightGBM forecaster for one metric.

    Uses temporal train/test split: last test_fraction of samples held out.
    Reports MAE on test set for reliable performance estimate.

    Returns:
        Dict with model_path, train_mae, test_mae, samples, or None if insufficient data
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm_not_installed", message="pip install lightgbm")
        return None

    end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=min_days + lookback_days + projection_days)).strftime(
        "%Y-%m-%d"
    )

    metrics = storage.read_gold_metrics(
        metric_names=[metric_name],
        start_date=start_date,
        end_date=end_date,
    )
    if len(metrics) < min_days:
        logger.warning(
            "insufficient_data",
            metric=metric_name,
            got=len(metrics),
            required=min_days,
        )
        return None

    by_date = {}
    for m in metrics:
        d = m.get("metric_date")
        v = m.get("metric_value")
        if d and v is not None:
            by_date[d] = float(v)

    dates_sorted = sorted(by_date.keys())
    values = [by_date[d] for d in dates_sorted]
    n = len(values)
    n_lags = lookback_days

    X_list, y_list = [], []
    for i in range(n_lags + projection_days, n):
        window = values[i - n_lags - projection_days : i - projection_days]
        target = values[i]
        feat = build_lag_features(window, n_lags)
        if feat:
            X_list.append(feat)
            y_list.append(target)

    if len(X_list) < 10:
        logger.warning("insufficient_training_samples", metric=metric_name, samples=len(X_list))
        return None

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    # Handle NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(X) < 10:
        return None

    # Temporal train/test split (last test_fraction for test — no shuffling)
    n_total = len(X)
    n_test = max(1, int(n_total * test_fraction))
    n_train = n_total - n_test
    if n_train < 10 or n_test < 3:
        n_test = 0
        n_train = n_total
        X_train, y_train = X, y
        X_test, y_test = np.array([]), np.array([])
    else:
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    # Strong regularization to reduce overfitting on the small time-series training sets.
    # max_depth=3 / num_leaves=7 limits tree complexity; min_child_samples=15 avoids
    # fitting tiny leaf nodes; subsample/colsample add stochasticity for generalisation.
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.02,
        min_child_samples=15,
        reg_alpha=0.1,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    if len(X_test) >= 3:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
    else:
        model.fit(X_train, y_train)

    train_mae = float(np.mean(np.abs(model.predict(X_train) - y_train)))
    test_mae = (
        float(np.mean(np.abs(model.predict(X_test) - y_test)))
        if len(X_test) > 0
        else None
    )

    logger.info(
        "metric_forecaster_trained",
        metric=metric_name,
        train_samples=n_train,
        test_samples=n_test,
        train_mae=round(train_mae, 6),
        test_mae=round(test_mae, 6) if test_mae is not None else None,
    )

    models_dir = models_dir or Path(get_settings().db_path).parent.parent / "models" / "trend"
    models_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "metric_name": metric_name,
        "lookback_days": lookback_days,
        "projection_days": projection_days,
        "n_features": X_train.shape[1],
        "n_lags": n_lags,
        "trained_at": datetime.utcnow().isoformat(),
        "train_mae": train_mae,
        "test_mae": test_mae,
        "mae": test_mae if test_mae is not None else train_mae,  # backward compat
        "train_samples": n_train,
        "test_samples": n_test,
        "samples": n_train,  # backward compat for consumers that expect "samples"
    }
    model_path = models_dir / f"forecaster_{metric_name}.joblib"
    joblib.dump({"model": model, "meta": meta}, model_path)
    return {
        "model_path": str(model_path),
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_samples": n_train,
        "test_samples": n_test,
    }


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM trend forecasters for Gold metrics")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Lookback days for features")
    parser.add_argument("--projection", type=int, default=DEFAULT_PROJECTION, help="Days ahead to project")
    parser.add_argument("--min-days", type=int, default=DEFAULT_MIN_TRAINING_DAYS, help="Min training days")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fraction of samples held out for test (temporal split, default 0.2)",
    )
    parser.add_argument("--models-dir", type=str, default=None, help="Output directory for models")
    args = parser.parse_args()

    settings = get_settings()
    storage = DuckDBStorage(db_path=settings.db_path)
    models_dir = (
        Path(args.models_dir)
        if args.models_dir
        else Path(settings.db_path).resolve().parent.parent / "models" / "trend"
    )

    print("\n" + "=" * 60)
    print("LedgerGuard — Trend Forecaster Training")
    print("=" * 60)
    print(f"Lookback: {args.lookback}d  Projection: {args.projection}d  Min days: {args.min_days}")
    print(f"Test fraction: {args.test_fraction} (temporal split)")
    print(f"Models dir: {models_dir}")
    print("=" * 60 + "\n")

    results = []
    for metric_name in TREND_THRESHOLDS:
        r = train_forecaster(
            storage=storage,
            metric_name=metric_name,
            lookback_days=args.lookback,
            projection_days=args.projection,
            min_days=args.min_days,
            test_fraction=args.test_fraction,
            models_dir=models_dir,
        )
        if r:
            results.append((metric_name, r))
            if r.get("test_mae") is not None:
                print(
                    f"  [OK] {metric_name}: train MAE={r['train_mae']:.6f} ({r['train_samples']} train)  "
                    f"test MAE={r['test_mae']:.6f} ({r['test_samples']} test)"
                )
            else:
                print(f"  [OK] {metric_name}: train MAE={r['train_mae']:.6f} ({r['train_samples']} samples)")
        else:
            print(f"  [SKIP] {metric_name}: insufficient data")

    if not results:
        print("\n  No models trained. Ensure Gold metrics exist:")
        print("    python scripts/seed_sandbox.py --mode local")
        print("  Then re-run this script.\n")
        sys.exit(1)

    print(f"\n  Trained {len(results)}/{len(TREND_THRESHOLDS)} forecasters -> {models_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
