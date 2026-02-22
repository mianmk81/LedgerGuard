"""
LedgerGuard — Master Training Orchestrator
============================================
Trains all ML models used by the Business Reliability Engine.

Usage:
    python scripts/train_models.py                    # Train all models (13 total)
    python scripts/train_models.py --model anomaly    # Train anomaly detection (4 models)
    python scripts/train_models.py --model churn      # Train churn prediction (3 models)
    python scripts/train_models.py --model delivery   # Train late delivery risk (3 models)
    python scripts/train_models.py --model sentiment  # Train sentiment analysis (3 models)

Requirements:
    pip install pandas scikit-learn lightgbm xgboost mlflow joblib matplotlib
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

PIPELINES = {
    "anomaly": {
        "script": "scripts/train_anomaly_detector.py",
        "description": "Anomaly Detection (Isolation Forest, One-Class SVM, LOF, Autoencoder)",
        "model_count": 4,
    },
    "churn": {
        "script": "scripts/train_churn_model.py",
        "description": "Churn Prediction (LightGBM, Logistic Regression, Random Forest)",
        "model_count": 3,
    },
    "delivery": {
        "script": "scripts/train_late_delivery.py",
        "description": "Late Delivery Risk (XGBoost, Random Forest, Logistic Regression)",
        "model_count": 3,
    },
    "sentiment": {
        "script": "scripts/train_sentiment.py",
        "description": "Sentiment Analysis (TF-IDF + LogReg, Naive Bayes, Random Forest)",
        "model_count": 3,
    },
}


def run_pipeline(name: str, info: dict) -> dict:
    """Run a single training pipeline and capture results."""
    script_path = PROJECT_ROOT / info["script"]
    if not script_path.exists():
        return {"status": "skipped", "error": f"Script not found: {script_path}"}

    print(f"\n{'='*70}")
    print(f"  TRAINING: {info['description']}")
    print(f"  Script:   {info['script']}")
    print(f"  Models:   {info['model_count']}")
    print(f"{'='*70}\n")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--model", "all"],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
            timeout=600,
        )
        duration = time.time() - start
        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration_seconds": round(duration, 1),
            "models_trained": info["model_count"],
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Exceeded 10 minute timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="LedgerGuard Master Training Orchestrator")
    parser.add_argument(
        "--model",
        choices=["anomaly", "churn", "delivery", "sentiment", "all"],
        default="all",
        help="Which model pipeline to train (default: all)",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipelines_to_run = (
        PIPELINES if args.model == "all" else {args.model: PIPELINES[args.model]}
    )

    total_models = sum(p["model_count"] for p in pipelines_to_run.values())
    print(f"\nLedgerGuard Training Orchestrator")
    print(f"{'='*70}")
    print(f"  Pipelines : {len(pipelines_to_run)}")
    print(f"  Models    : {total_models}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    overall_start = time.time()
    results = {}

    for name, info in pipelines_to_run.items():
        results[name] = run_pipeline(name, info)

    overall_duration = time.time() - overall_start

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Pipeline':<15} {'Status':<10} {'Models':<8} {'Duration':<12}")
    print(f"  {'-'*45}")

    total_trained = 0
    for name, result in results.items():
        status = result["status"]
        models = result.get("models_trained", 0) if status == "success" else 0
        duration = f"{result.get('duration_seconds', 0)}s"
        total_trained += models
        print(f"  {name:<15} {status:<10} {models:<8} {duration:<12}")

    print(f"  {'-'*45}")
    print(f"  Total: {total_trained}/{total_models} models trained in {overall_duration:.1f}s")
    print(f"{'='*70}\n")

    # Save report
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_duration_seconds": round(overall_duration, 1),
        "total_models_trained": total_trained,
        "total_models_expected": total_models,
        "pipelines": results,
    }
    report_path = REPORTS_DIR / "training_summary.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {report_path}")


if __name__ == "__main__":
    main()
