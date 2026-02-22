"""
Anomaly Detection Model Training Script for LedgerGuard.

Trains and evaluates 4 anomaly detection models on Olist e-commerce data:
1. Isolation Forest
2. One-Class SVM
3. Local Outlier Factor
4. Autoencoder (MLPRegressor-based)

All experiments tracked with MLflow, models saved to models/anomaly/ directory.
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_loader import OlistDataLoader

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# MLflow experiment name
MLFLOW_EXPERIMENT = "ledgerguard-anomaly-detection"

# Model save directory
MODELS_DIR = Path(__file__).parent.parent / "models" / "anomaly"


def setup_mlflow():
    """Initialize MLflow experiment."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info("mlflow_experiment_set", experiment=MLFLOW_EXPERIMENT)


def create_pseudo_labels(scores: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Create pseudo ground-truth labels for evaluation.

    Since we don't have true anomaly labels, we treat the top X% most anomalous
    samples (based on anomaly scores) as anomalies for evaluation purposes.

    Args:
        scores: Anomaly scores (higher = more anomalous)
        contamination: Fraction to treat as anomalies

    Returns:
        Binary labels (1 = anomaly, 0 = normal)
    """
    threshold = np.percentile(scores, (1 - contamination) * 100)
    return (scores >= threshold).astype(int)


def evaluate_model(
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
    dataset_name: str = "test",
    contamination: float = 0.05
) -> Dict[str, float]:
    """
    Evaluate anomaly detection model using pseudo-labels.

    Args:
        y_pred: Binary predictions (-1 for anomaly, 1 for normal)
        anomaly_scores: Continuous anomaly scores
        dataset_name: Name of dataset being evaluated
        contamination: Expected contamination rate

    Returns:
        Dictionary of evaluation metrics
    """
    # Convert predictions to binary (1 = anomaly, 0 = normal)
    y_pred_binary = (y_pred == -1).astype(int)

    # Create pseudo ground truth
    y_true = create_pseudo_labels(anomaly_scores, contamination)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary', zero_division=0
    )

    # Calculate AUC if we have both classes
    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        auc = 0.0

    # Count anomalies
    n_anomalies = np.sum(y_pred_binary)
    anomaly_rate = n_anomalies / len(y_pred_binary)

    metrics = {
        f"{dataset_name}_precision": precision,
        f"{dataset_name}_recall": recall,
        f"{dataset_name}_f1": f1,
        f"{dataset_name}_auc": auc,
        f"{dataset_name}_anomalies": int(n_anomalies),
        f"{dataset_name}_anomaly_rate": anomaly_rate,
    }

    logger.info(
        f"evaluation_complete_{dataset_name}",
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        anomalies=n_anomalies,
    )

    return metrics


def print_evaluation_report(
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
    dataset_name: str = "Test"
):
    """Print detailed evaluation report."""
    y_pred_binary = (y_pred == -1).astype(int)
    y_true = create_pseudo_labels(anomaly_scores, contamination=0.05)

    print(f"\n{'=' * 70}")
    print(f"{dataset_name} Set Evaluation")
    print(f"{'=' * 70}")

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred_binary,
        target_names=['Normal', 'Anomaly'],
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_binary)
    print(f"{'':>15} Predicted Normal  Predicted Anomaly")
    print(f"Actual Normal   {cm[0, 0]:>15}  {cm[0, 1]:>17}")
    print(f"Actual Anomaly  {cm[1, 0]:>15}  {cm[1, 1]:>17}")

    print(f"\nAnomaly Statistics:")
    print(f"  Total samples: {len(y_pred)}")
    print(f"  Detected anomalies: {np.sum(y_pred_binary)} ({np.mean(y_pred_binary)*100:.2f}%)")
    print(f"  Score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    print(f"  Score mean: {anomaly_scores.mean():.4f}")
    print(f"  Score std: {anomaly_scores.std():.4f}")


def train_isolation_forest(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[IsolationForest, Dict[str, float]]:
    """
    Train Isolation Forest anomaly detector.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features

    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    logger.info("training_isolation_forest")

    with mlflow.start_run(run_name="isolation_forest"):
        # Define hyperparameters
        params = {
            'n_estimators': 200,
            'contamination': 0.02,  # Stricter: expect ~2% anomalies
            'max_features': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        }

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "isolation_forest")

        # Train model
        start_time = datetime.now()
        model = IsolationForest(**params)
        model.fit(X_train)
        training_time = (datetime.now() - start_time).total_seconds()

        mlflow.log_metric("training_time_seconds", training_time)

        # Predict and evaluate
        metrics = {}

        # Training set
        y_train_pred = model.predict(X_train)
        train_scores = -model.score_samples(X_train)  # Negative for higher = more anomalous
        train_metrics = evaluate_model(y_train_pred, train_scores, "train")
        metrics.update(train_metrics)

        # Validation set
        y_val_pred = model.predict(X_val)
        val_scores = -model.score_samples(X_val)
        val_metrics = evaluate_model(y_val_pred, val_scores, "val")
        metrics.update(val_metrics)

        # Test set
        y_test_pred = model.predict(X_test)
        test_scores = -model.score_samples(X_test)
        test_metrics = evaluate_model(y_test_pred, test_scores, "test")
        metrics.update(test_metrics)

        # Log all metrics
        mlflow.log_metrics(metrics)

        # Save model
        model_path = MODELS_DIR / "isolation_forest.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")

        logger.info(
            "isolation_forest_trained",
            training_time=training_time,
            test_f1=metrics['test_f1'],
            model_path=str(model_path),
        )

        # Print detailed report
        print_evaluation_report(y_test_pred, test_scores, "Isolation Forest - Test")

        return model, metrics


def train_one_class_svm(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[Tuple[OneClassSVM, StandardScaler], Dict[str, float]]:
    """
    Train One-Class SVM anomaly detector.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features

    Returns:
        Tuple of ((trained model, scaler), metrics dictionary)
    """
    logger.info("training_one_class_svm")

    with mlflow.start_run(run_name="one_class_svm"):
        # Define hyperparameters
        params = {
            'kernel': 'rbf',
            'gamma': 'auto',
            'nu': 0.02,  # Stricter: expect ~2% outliers
        }

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "one_class_svm")

        # Scale features (critical for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        start_time = datetime.now()
        model = OneClassSVM(**params)
        model.fit(X_train_scaled)
        training_time = (datetime.now() - start_time).total_seconds()

        mlflow.log_metric("training_time_seconds", training_time)

        # Predict and evaluate
        metrics = {}

        # Training set
        y_train_pred = model.predict(X_train_scaled)
        train_scores = -model.score_samples(X_train_scaled)
        train_metrics = evaluate_model(y_train_pred, train_scores, "train")
        metrics.update(train_metrics)

        # Validation set
        y_val_pred = model.predict(X_val_scaled)
        val_scores = -model.score_samples(X_val_scaled)
        val_metrics = evaluate_model(y_val_pred, val_scores, "val")
        metrics.update(val_metrics)

        # Test set
        y_test_pred = model.predict(X_test_scaled)
        test_scores = -model.score_samples(X_test_scaled)
        test_metrics = evaluate_model(y_test_pred, test_scores, "test")
        metrics.update(test_metrics)

        # Log all metrics
        mlflow.log_metrics(metrics)

        # Save model and scaler
        model_path = MODELS_DIR / "one_class_svm.joblib"
        scaler_path = MODELS_DIR / "one_class_svm_scaler.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        mlflow.sklearn.log_model(model, "model")

        logger.info(
            "one_class_svm_trained",
            training_time=training_time,
            test_f1=metrics['test_f1'],
            model_path=str(model_path),
        )

        # Print detailed report
        print_evaluation_report(y_test_pred, test_scores, "One-Class SVM - Test")

        return (model, scaler), metrics


def train_local_outlier_factor(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[Tuple[LocalOutlierFactor, StandardScaler], Dict[str, float]]:
    """
    Train Local Outlier Factor anomaly detector.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features

    Returns:
        Tuple of ((trained model, scaler), metrics dictionary)
    """
    logger.info("training_local_outlier_factor")

    with mlflow.start_run(run_name="local_outlier_factor"):
        # Define hyperparameters
        params = {
            'n_neighbors': 20,
            'contamination': 0.02,  # Stricter: expect ~2% anomalies
            'novelty': True,  # Enable novelty detection for predict()
            'n_jobs': -1,
        }

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "local_outlier_factor")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        start_time = datetime.now()
        model = LocalOutlierFactor(**params)
        model.fit(X_train_scaled)
        training_time = (datetime.now() - start_time).total_seconds()

        mlflow.log_metric("training_time_seconds", training_time)

        # Predict and evaluate
        metrics = {}

        # Training set (use decision_function for training data)
        y_train_pred = model.predict(X_train_scaled)
        train_scores = -model.score_samples(X_train_scaled)
        train_metrics = evaluate_model(y_train_pred, train_scores, "train")
        metrics.update(train_metrics)

        # Validation set
        y_val_pred = model.predict(X_val_scaled)
        val_scores = -model.score_samples(X_val_scaled)
        val_metrics = evaluate_model(y_val_pred, val_scores, "val")
        metrics.update(val_metrics)

        # Test set
        y_test_pred = model.predict(X_test_scaled)
        test_scores = -model.score_samples(X_test_scaled)
        test_metrics = evaluate_model(y_test_pred, test_scores, "test")
        metrics.update(test_metrics)

        # Log all metrics
        mlflow.log_metrics(metrics)

        # Save model and scaler
        model_path = MODELS_DIR / "local_outlier_factor.joblib"
        scaler_path = MODELS_DIR / "local_outlier_factor_scaler.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        mlflow.sklearn.log_model(model, "model")

        logger.info(
            "local_outlier_factor_trained",
            training_time=training_time,
            test_f1=metrics['test_f1'],
            model_path=str(model_path),
        )

        # Print detailed report
        print_evaluation_report(y_test_pred, test_scores, "Local Outlier Factor - Test")

        return (model, scaler), metrics


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[Tuple[MLPRegressor, StandardScaler, float], Dict[str, float]]:
    """
    Train Autoencoder-based anomaly detector using MLPRegressor.

    The autoencoder learns to reconstruct normal patterns. Anomalies are
    detected based on high reconstruction error. Uses validation-based
    threshold tuning to reduce overflagging and improve precision.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features

    Returns:
        Tuple of ((trained model, scaler, threshold), metrics dictionary)
    """
    logger.info("training_autoencoder")

    with mlflow.start_run(run_name="autoencoder"):
        # Define hyperparameters - wider bottleneck to handle windowed input
        n_features = X_train.shape[1]
        params = {
            'hidden_layer_sizes': (128, 64, 32, 64, 128),  # Symmetric bottleneck
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,  # L2 regularization to avoid overfitting
            'max_iter': 1000,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 25,
        }

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "autoencoder")
        mlflow.log_param("n_features", n_features)

        # Scale features (StandardScaler works better for reconstruction threshold tuning)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Build sliding windows over the scaled time-series data
        WINDOW_SIZE = 7

        def build_windows(X: np.ndarray, w: int = WINDOW_SIZE) -> np.ndarray:
            """Flatten consecutive windows of length w into single rows."""
            if len(X) <= w:
                return X  # Not enough data, fall back to flat
            rows = []
            for i in range(w, len(X)):
                window = X[i - w:i].flatten()
                rows.append(window)
            return np.array(rows)

        X_train_windowed = build_windows(X_train_scaled)
        X_val_windowed = build_windows(X_val_scaled)
        X_test_windowed = build_windows(X_test_scaled)

        mlflow.log_param("window_size", WINDOW_SIZE)
        mlflow.log_param("train_windowed_shape", str(X_train_windowed.shape))

        # Train autoencoder on windowed data (reconstruct windowed input)
        start_time = datetime.now()
        model = MLPRegressor(**params)
        model.fit(X_train_windowed, X_train_windowed)  # Train to reconstruct windowed input
        training_time = (datetime.now() - start_time).total_seconds()

        mlflow.log_metric("training_time_seconds", training_time)

        # Calculate reconstruction errors on windowed data
        def get_reconstruction_error(X: np.ndarray) -> np.ndarray:
            """Calculate mean squared error between windowed input and reconstruction."""
            X_reconstructed = model.predict(X)
            mse = np.mean((X - X_reconstructed) ** 2, axis=1)
            return mse

        # Training set reconstruction errors (windowed)
        train_errors = get_reconstruction_error(X_train_windowed)
        val_errors = get_reconstruction_error(X_val_windowed)

        # Tune threshold: prioritize precision (avoid overflagging).
        # Target ~2% anomaly rate; try percentiles 98-99.9.
        target_rate = 0.02
        best_score = -1
        best_threshold = np.percentile(train_errors, 99.5)
        for p in [98, 99, 99.5, 99.9]:
            thr = np.percentile(train_errors, p)
            y_val_pred = np.where(val_errors > thr, -1, 1)
            val_anomaly_rate = np.mean(y_val_pred == -1)
            # Score: prefer anomaly rate near target; penalize overflagging
            rate_penalty = abs(val_anomaly_rate - target_rate)
            y_val_binary = (y_val_pred == -1).astype(int)
            y_true = create_pseudo_labels(val_errors, contamination=target_rate)
            _, prec, f1, _ = precision_recall_fscore_support(
                y_true, y_val_binary, average='binary', zero_division=0
            )
            # Combine: favor precision and rate close to target
            score = prec - 2 * rate_penalty + 0.5 * f1
            if score > best_score:
                best_score = score
                best_threshold = thr

        threshold = best_threshold
        mlflow.log_param("reconstruction_threshold", threshold)
        mlflow.log_param("threshold_tuning", "precision_and_rate")

        # Predict and evaluate using windowed data
        metrics = {}

        # Training set
        train_scores = train_errors
        y_train_pred = np.where(train_scores > threshold, -1, 1)
        train_metrics = evaluate_model(y_train_pred, train_scores, "train")
        metrics.update(train_metrics)

        # Validation set
        val_errors = get_reconstruction_error(X_val_windowed)
        val_scores = val_errors
        y_val_pred = np.where(val_scores > threshold, -1, 1)
        val_metrics = evaluate_model(y_val_pred, val_scores, "val")
        metrics.update(val_metrics)

        # Test set
        test_errors = get_reconstruction_error(X_test_windowed)
        test_scores = test_errors
        y_test_pred = np.where(test_scores > threshold, -1, 1)
        test_metrics = evaluate_model(y_test_pred, test_scores, "test")
        metrics.update(test_metrics)

        # Log error statistics
        mlflow.log_metrics({
            "train_error_mean": train_errors.mean(),
            "train_error_std": train_errors.std(),
            "val_error_mean": val_errors.mean(),
            "val_error_std": val_errors.std(),
            "test_error_mean": test_errors.mean(),
            "test_error_std": test_errors.std(),
        })

        # Log all metrics
        mlflow.log_metrics(metrics)

        # Save model, scaler, and threshold
        model_path = MODELS_DIR / "autoencoder.joblib"
        scaler_path = MODELS_DIR / "autoencoder_scaler.joblib"
        threshold_path = MODELS_DIR / "autoencoder_threshold.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(threshold_path, 'w') as f:
            f.write(str(threshold))

        mlflow.sklearn.log_model(model, "model")

        logger.info(
            "autoencoder_trained",
            training_time=training_time,
            test_f1=metrics['test_f1'],
            threshold=threshold,
            model_path=str(model_path),
        )

        # Print detailed report
        print_evaluation_report(y_test_pred, test_scores, "Autoencoder - Test")

        return (model, scaler, threshold), metrics


def train_ensemble(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    min_votes: int = 2,
) -> Dict[str, float]:
    """
    Ensemble: flag anomaly only if at least min_votes models agree.
    Loads trained IF, OCSVM, LOF, Autoencoder from disk.

    Args:
        X_train, X_val, X_test: Feature arrays
        min_votes: Minimum number of models that must flag anomaly (default 2)

    Returns:
        Metrics dictionary
    """
    logger.info("training_ensemble", min_votes=min_votes)

    with mlflow.start_run(run_name="ensemble"):
        mlflow.log_param("model_type", "ensemble")
        mlflow.log_param("min_votes", min_votes)
        mlflow.log_param("base_models", ["isolation_forest", "ocsvm", "lof", "autoencoder"])

        predictions = {"train": [], "val": [], "test": []}

        # Load and predict with each model
        # 1. Isolation Forest (no scaler)
        if_path = MODELS_DIR / "isolation_forest.joblib"
        if if_path.exists():
            if_model = joblib.load(if_path)
            for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
                pred = if_model.predict(X)
                predictions[split].append(pred)
        else:
            logger.warning("ensemble_skip_model", model="isolation_forest", reason="not_found")

        # 2. One-Class SVM (needs scaler)
        ocsvm_path = MODELS_DIR / "one_class_svm.joblib"
        ocsvm_scaler_path = MODELS_DIR / "one_class_svm_scaler.joblib"
        if ocsvm_path.exists() and ocsvm_scaler_path.exists():
            ocsvm_model = joblib.load(ocsvm_path)
            ocsvm_scaler = joblib.load(ocsvm_scaler_path)
            for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
                X_s = ocsvm_scaler.transform(X)
                pred = ocsvm_model.predict(X_s)
                predictions[split].append(pred)
        else:
            logger.warning("ensemble_skip_model", model="one_class_svm", reason="not_found")

        # 3. LOF (needs scaler)
        lof_path = MODELS_DIR / "local_outlier_factor.joblib"
        lof_scaler_path = MODELS_DIR / "local_outlier_factor_scaler.joblib"
        if lof_path.exists() and lof_scaler_path.exists():
            lof_model = joblib.load(lof_path)
            lof_scaler = joblib.load(lof_scaler_path)
            for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
                X_s = lof_scaler.transform(X)
                pred = lof_model.predict(X_s)
                predictions[split].append(pred)
        else:
            logger.warning("ensemble_skip_model", model="lof", reason="not_found")

        # 4. Autoencoder (needs scaler + threshold; may be windowed)
        ae_path = MODELS_DIR / "autoencoder.joblib"
        ae_scaler_path = MODELS_DIR / "autoencoder_scaler.joblib"
        ae_threshold_path = MODELS_DIR / "autoencoder_threshold.txt"
        if ae_path.exists() and ae_scaler_path.exists() and ae_threshold_path.exists():
            ae_model = joblib.load(ae_path)
            ae_scaler = joblib.load(ae_scaler_path)
            with open(ae_threshold_path) as f:
                ae_threshold = float(f.read().strip())

            # Check if AE expects windowed input
            ae_n_features = ae_model.n_features_in_
            WINDOW_SIZE = 7

            for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
                X_s = ae_scaler.transform(X)
                if ae_n_features > X_s.shape[1]:
                    # Windowed AE: build sliding windows
                    pred = np.ones(len(X_s))  # default normal
                    for i in range(WINDOW_SIZE, len(X_s)):
                        window = X_s[i - WINDOW_SIZE:i].flatten().reshape(1, -1)
                        # Pad/truncate to match expected input
                        if window.shape[1] < ae_n_features:
                            pad = np.zeros((1, ae_n_features - window.shape[1]))
                            window = np.concatenate([window, pad], axis=1)
                        elif window.shape[1] > ae_n_features:
                            window = window[:, :ae_n_features]
                        recon = ae_model.predict(window)
                        err = float(np.mean((window - recon) ** 2))
                        if err > ae_threshold:
                            pred[i] = -1
                    predictions[split].append(pred)
                else:
                    # Flat AE (legacy)
                    recon = ae_model.predict(X_s)
                    errors = np.mean((X_s - recon) ** 2, axis=1)
                    pred = np.where(errors > ae_threshold, -1, 1)
                    predictions[split].append(pred)
        else:
            logger.warning("ensemble_skip_model", model="autoencoder", reason="not_found")

        n_models = len(predictions["train"])
        if n_models < 2:
            logger.error("ensemble_insufficient_models", n_models=n_models)
            return {}

        # Vote: anomaly (-1) if at least min_votes models say anomaly
        def vote(arrs: List[np.ndarray]) -> np.ndarray:
            stacked = np.column_stack(arrs)
            n_anomaly_votes = np.sum(stacked == -1, axis=1)
            return np.where(n_anomaly_votes >= min_votes, -1, 1)

        y_train_ens = vote(predictions["train"])
        y_val_ens = vote(predictions["val"])
        y_test_ens = vote(predictions["test"])

        # Use average of model scores as ensemble score (simplified: use vote count)
        def ensemble_scores(arrs: List[np.ndarray]) -> np.ndarray:
            stacked = np.column_stack(arrs)
            # Score = fraction of models that said anomaly (higher = more anomalous)
            return np.mean(stacked == -1, axis=1)

        train_scores = ensemble_scores(predictions["train"])
        val_scores = ensemble_scores(predictions["val"])
        test_scores = ensemble_scores(predictions["test"])

        metrics = {}
        for name, y_pred, scores in [
            ("train", y_train_ens, train_scores),
            ("val", y_val_ens, val_scores),
            ("test", y_test_ens, test_scores),
        ]:
            m = evaluate_model(y_pred, scores, name, contamination=0.02)
            metrics.update(m)

        mlflow.log_metrics(metrics)

        # Save ensemble config
        config = {"min_votes": min_votes, "n_models": n_models}
        joblib.dump(config, MODELS_DIR / "ensemble_config.joblib")

        logger.info(
            "ensemble_trained",
            n_models=n_models,
            test_f1=metrics["test_f1"],
            test_anomaly_rate=metrics["test_anomaly_rate"],
        )

        print_evaluation_report(y_test_ens, test_scores, "Ensemble - Test")

        return metrics


def print_comparison_table(all_results: Dict[str, Dict[str, float]]):
    """Print comparison table of all models."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)

    # Define metrics to compare
    metrics_to_compare = ['test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_anomaly_rate']

    # Create DataFrame for easy display
    comparison_data = []
    for model_name, metrics in all_results.items():
        row = {'Model': model_name}
        for metric in metrics_to_compare:
            row[metric] = metrics.get(metric, 0.0)
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Format and print
    print("\nTest Set Performance:")
    print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12} {'Anomaly %':<12}")
    print("-" * 100)

    for _, row in df.iterrows():
        print(
            f"{row['Model']:<25} "
            f"{row['test_precision']:<12.4f} "
            f"{row['test_recall']:<12.4f} "
            f"{row['test_f1']:<12.4f} "
            f"{row['test_auc']:<12.4f} "
            f"{row['test_anomaly_rate']*100:<12.2f}"
        )

    # Find best model by F1 score
    best_idx = df['test_f1'].idxmax()
    best_model = df.loc[best_idx, 'Model']
    best_f1 = df.loc[best_idx, 'test_f1']

    print("\n" + "=" * 100)
    print(f"BEST MODEL: {best_model} (F1-Score: {best_f1:.4f})")
    print("=" * 100 + "\n")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models on Olist e-commerce data"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['isolation_forest', 'ocsvm', 'lof', 'autoencoder', 'ensemble', 'all'],
        help='Model to train (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory containing Olist CSV files'
    )

    args = parser.parse_args()

    print("\n" + "=" * 100)
    print("LEDGERGUARD ANOMALY DETECTION MODEL TRAINING")
    print("=" * 100)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to train: {args.model}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")
    print(f"Model save directory: {MODELS_DIR}")
    print("=" * 100 + "\n")

    # Setup MLflow
    setup_mlflow()

    # Load data
    logger.info("loading_data", data_dir=args.data_dir)
    print("\n[1/5] Loading and preparing data...")

    data_loader = OlistDataLoader(data_dir=args.data_dir)

    try:
        X_train, X_val, X_test, dates_train, dates_val, dates_test = \
            data_loader.prepare_anomaly_detection_data()
    except FileNotFoundError as e:
        logger.error("data_loading_failed", error=str(e))
        print(f"\nERROR: {e}")
        print("\nPlease download the Olist dataset from:")
        print("https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
        print(f"\nAnd place the CSV files in: {data_loader.data_dir}")
        return 1

    print(f"  Train samples: {len(X_train)} (dates: {dates_train.min()} to {dates_train.max()})")
    print(f"  Val samples: {len(X_val)} (dates: {dates_val.min()} to {dates_val.max()})")
    print(f"  Test samples: {len(X_test)} (dates: {dates_test.min()} to {dates_test.max()})")
    print(f"  Features: {X_train.shape[1]}")

    # Train models
    all_results = {}
    models_to_train = []

    if args.model == 'all':
        models_to_train = ['isolation_forest', 'ocsvm', 'lof', 'autoencoder', 'ensemble']
    elif args.model == 'ensemble':
        models_to_train = ['ensemble']
    else:
        models_to_train = [args.model]

    for i, model_name in enumerate(models_to_train, start=2):
        print(f"\n[{i}/5] Training {model_name.replace('_', ' ').title()}...")

        try:
            if model_name == 'isolation_forest':
                _, metrics = train_isolation_forest(X_train, X_val, X_test)
                all_results['Isolation Forest'] = metrics

            elif model_name == 'ocsvm':
                _, metrics = train_one_class_svm(X_train, X_val, X_test)
                all_results['One-Class SVM'] = metrics

            elif model_name == 'lof':
                _, metrics = train_local_outlier_factor(X_train, X_val, X_test)
                all_results['Local Outlier Factor'] = metrics

            elif model_name == 'autoencoder':
                _, metrics = train_autoencoder(X_train, X_val, X_test)
                all_results['Autoencoder'] = metrics

            elif model_name == 'ensemble':
                metrics = train_ensemble(X_train, X_val, X_test, min_votes=3)
                if metrics:
                    all_results['Ensemble (3+ vote)'] = metrics

        except Exception as e:
            logger.error("model_training_failed", model=model_name, error=str(e))
            print(f"  ERROR: Failed to train {model_name}: {e}")
            continue

    # Print comparison
    if len(all_results) > 1:
        print(f"\n[{len(models_to_train)+2}/5] Comparing models...")
        print_comparison_table(all_results)

    # Summary
    print("\n" + "=" * 100)
    print("TRAINING COMPLETE")
    print("=" * 100)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models trained: {len(all_results)}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"MLflow tracking: {mlflow.get_tracking_uri()}")
    print("\nTo view results in MLflow UI:")
    print("  mlflow ui")
    print("=" * 100 + "\n")

    logger.info(
        "training_pipeline_complete",
        models_trained=len(all_results),
        models_dir=str(MODELS_DIR),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
