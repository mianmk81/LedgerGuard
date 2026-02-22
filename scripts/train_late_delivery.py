#!/usr/bin/env python3
"""
Late Delivery Prediction Model Training.

Trains multiple ML models to predict late deliveries on Olist e-commerce data:
- XGBoost Classifier (gradient boosting)
- Random Forest (ensemble decision trees)
- Logistic Regression (linear baseline)
- Two-Stage Model (regression + classification)
- Stacked Ensemble (XGB + RF + LR meta-learner)

Each model:
- Trains on training set
- Optimizes decision threshold on validation set (Optuna or PR-curve)
- Evaluates on test set with comprehensive metrics
- Tracked with MLflow for experiment management
- Saved to models/delivery/ for deployment

Usage:
    python scripts/train_late_delivery.py [--model MODEL]

    MODEL can be: xgboost, random_forest, logistic, two_stage, ensemble, or all (default)

Example:
    python scripts/train_late_delivery.py --model xgboost
    python scripts/train_late_delivery.py --model two_stage
    python scripts/train_late_delivery.py  # trains all models
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loader import OlistDataLoader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = structlog.get_logger()


class LateDeliveryTrainer:
    """
    Late delivery prediction model trainer.

    Handles training, threshold optimization, evaluation, and persistence
    for multiple classification models.
    """

    def __init__(self, models_dir: str = "models/delivery", mlflow_experiment: str = "ledgerguard-late-delivery"):
        """
        Initialize trainer.

        Args:
            models_dir: Directory to save trained models
            mlflow_experiment: MLflow experiment name
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.mlflow_experiment = mlflow_experiment
        mlflow.set_experiment(mlflow_experiment)

        self.logger = logger.bind(component="late_delivery_trainer")
        self.results = []

    def _log_timestamp(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _apply_smote(
        self, X: pd.DataFrame, y: pd.Series, sampling_strategy: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE oversampling to balance classes.

        Args:
            X: Training features
            y: Training labels
            sampling_strategy: Target ratio of minority/majority (0.5 = 1:2 ratio)

        Returns:
            Resampled (X, y) with more minority samples
        """
        if not HAS_SMOTE:
            self.logger.warning("smote_not_available", reason="imbalanced-learn not installed")
            return X, y

        self._log_timestamp(
            f"Applying SMOTE: {y.mean():.2%} late -> ~{sampling_strategy:.0%} target ratio"
        )
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=5,
        )
        X_res, y_res = smote.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res, name=y.name)

        self.logger.info(
            "smote_applied",
            original_size=len(X),
            resampled_size=len(X_res),
            original_late_rate=f"{y.mean():.2%}",
            resampled_late_rate=f"{y_res.mean():.2%}",
        )
        return X_res, y_res

    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize decision threshold using F1 score on validation set.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        self._log_timestamp("Optimizing decision threshold on validation set")

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Calculate F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # Find threshold with best F1
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        metrics = {
            "threshold": best_threshold,
            "precision": precisions[best_idx],
            "recall": recalls[best_idx],
            "f1": best_f1,
        }

        self.logger.info(
            "threshold_optimized",
            threshold=f"{best_threshold:.4f}",
            f1=f"{best_f1:.4f}",
            precision=f"{metrics['precision']:.4f}",
            recall=f"{metrics['recall']:.4f}",
        )

        return best_threshold, metrics

    def _evaluate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
        set_name: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset with given threshold.

        Args:
            model: Trained model
            X: Features
            y: True labels
            threshold: Decision threshold
            set_name: Name of dataset (for logging)

        Returns:
            Dictionary of evaluation metrics
        """
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
        }

        # Calculate AUC-PR
        precision, recall, _ = precision_recall_curve(y, y_proba)
        metrics["pr_auc"] = auc(recall, precision)

        self.logger.info(
            f"{set_name}_evaluation",
            accuracy=f"{metrics['accuracy']:.4f}",
            precision=f"{metrics['precision']:.4f}",
            recall=f"{metrics['recall']:.4f}",
            f1=f"{metrics['f1']:.4f}",
            roc_auc=f"{metrics['roc_auc']:.4f}",
            pr_auc=f"{metrics['pr_auc']:.4f}",
        )

        return metrics, y_pred, y_proba

    def _print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Print detailed classification report."""
        print(f"\n{'='*80}")
        print(f"Classification Report - {model_name}")
        print(f"{'='*80}")
        print(classification_report(y_true, y_pred, target_names=["On-Time", "Late"], digits=4))

    def _print_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Print confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n{'='*80}")
        print(f"Confusion Matrix - {model_name}")
        print(f"{'='*80}")
        print(f"                  Predicted")
        print(f"                On-Time    Late")
        print(f"Actual On-Time    {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Late       {cm[1,0]:6d}  {cm[1,1]:6d}")
        print()

    def _print_feature_importances(self, model, feature_names: list, model_name: str, top_n: int = 15):
        """Print top N feature importances."""
        # Extract feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For logistic regression, use absolute coefficients
            importances = np.abs(model.coef_[0])
        elif isinstance(model, Pipeline):
            # For pipelines, get the final estimator
            final_estimator = model.steps[-1][1]
            if hasattr(final_estimator, "coef_"):
                importances = np.abs(final_estimator.coef_[0])
            elif hasattr(final_estimator, "feature_importances_"):
                importances = final_estimator.feature_importances_
            else:
                return
        else:
            return

        # Sort and get top N
        indices = np.argsort(importances)[::-1][:top_n]

        print(f"\n{'='*80}")
        print(f"Top {top_n} Feature Importances - {model_name}")
        print(f"{'='*80}")
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:8.6f}")
        print()

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[XGBClassifier, Dict[str, float], float]:
        """
        Train XGBoost classifier.

        Args:
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Label sets

        Returns:
            Tuple of (trained_model, test_metrics, optimal_threshold)
        """
        self._log_timestamp("Training XGBoost Classifier")

        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count

        self.logger.info(
            "xgboost_config",
            negative_samples=int(neg_count),
            positive_samples=int(pos_count),
            scale_pos_weight=f"{scale_pos_weight:.4f}",
        )

        with mlflow.start_run(run_name="xgboost"):
            # Define model
            model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                reg_alpha=0.1,
                reg_lambda=1.0,
            )

            # Log parameters
            mlflow.log_params({
                "model_type": "xgboost",
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 8,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
            })

            # Train model with early stopping
            self._log_timestamp("Fitting XGBoost model on training data")
            try:
                from xgboost.callback import EarlyStopping
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    callbacks=[EarlyStopping(rounds=50)],
                )
            except (ImportError, TypeError):
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

            # Optimize threshold on validation set
            y_val_proba = model.predict_proba(X_val)[:, 1]
            threshold, val_metrics = self._optimize_threshold(y_val, y_val_proba)

            # Evaluate on test set with optimized threshold
            self._log_timestamp("Evaluating XGBoost on test set")
            test_metrics, y_pred, y_proba = self._evaluate_model(model, X_test, y_test, threshold, "test")

            # Log metrics
            mlflow.log_metrics({
                "threshold": threshold,
                "val_f1": val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
            })

            # Save model
            model_path = self.models_dir / "xgboost_late_delivery.joblib"
            joblib.dump({"model": model, "threshold": threshold}, model_path)
            mlflow.log_artifact(str(model_path))
            self._log_timestamp(f"Model saved to {model_path}")

            # Print detailed results
            self._print_classification_report(y_test, y_pred, "XGBoost")
            self._print_confusion_matrix(y_test, y_pred, "XGBoost")
            self._print_feature_importances(model, X_train.columns.tolist(), "XGBoost", top_n=15)

            # Store results for comparison
            self.results.append({
                "model": "XGBoost",
                "threshold": threshold,
                **test_metrics,
                "y_test": y_test,
                "y_proba": y_proba,
            })

            return model, test_metrics, threshold

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[RandomForestClassifier, Dict[str, float], float]:
        """
        Train Random Forest classifier.

        Args:
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Label sets

        Returns:
            Tuple of (trained_model, test_metrics, optimal_threshold)
        """
        self._log_timestamp("Training Random Forest Classifier")

        with mlflow.start_run(run_name="random_forest"):
            # Define model
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )

            # Log parameters
            mlflow.log_params({
                "model_type": "random_forest",
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_leaf": 3,
                "class_weight": "balanced",
                "random_state": 42,
            })

            # Train model
            self._log_timestamp("Fitting Random Forest model on training data")
            model.fit(X_train, y_train)

            # Optimize threshold on validation set
            y_val_proba = model.predict_proba(X_val)[:, 1]
            threshold, val_metrics = self._optimize_threshold(y_val, y_val_proba)

            # Evaluate on test set with optimized threshold
            self._log_timestamp("Evaluating Random Forest on test set")
            test_metrics, y_pred, y_proba = self._evaluate_model(model, X_test, y_test, threshold, "test")

            # Log metrics
            mlflow.log_metrics({
                "threshold": threshold,
                "val_f1": val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
            })

            # Save model
            model_path = self.models_dir / "random_forest_late_delivery.joblib"
            joblib.dump({"model": model, "threshold": threshold}, model_path)
            mlflow.log_artifact(str(model_path))
            self._log_timestamp(f"Model saved to {model_path}")

            # Print detailed results
            self._print_classification_report(y_test, y_pred, "Random Forest")
            self._print_confusion_matrix(y_test, y_pred, "Random Forest")
            self._print_feature_importances(model, X_train.columns.tolist(), "Random Forest", top_n=15)

            # Store results for comparison
            self.results.append({
                "model": "Random Forest",
                "threshold": threshold,
                **test_metrics,
                "y_test": y_test,
                "y_proba": y_proba,
            })

            return model, test_metrics, threshold

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[Pipeline, Dict[str, float], float]:
        """
        Train Logistic Regression classifier with feature scaling.

        Args:
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Label sets

        Returns:
            Tuple of (trained_pipeline, test_metrics, optimal_threshold)
        """
        self._log_timestamp("Training Logistic Regression Classifier")

        with mlflow.start_run(run_name="logistic_regression"):
            # Define pipeline with scaling
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                    verbose=0,
                ))
            ])

            # Log parameters
            mlflow.log_params({
                "model_type": "logistic_regression",
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "scaled": True,
            })

            # Train model
            self._log_timestamp("Fitting Logistic Regression model on training data")
            model.fit(X_train, y_train)

            # Optimize threshold on validation set
            y_val_proba = model.predict_proba(X_val)[:, 1]
            threshold, val_metrics = self._optimize_threshold(y_val, y_val_proba)

            # Evaluate on test set with optimized threshold
            self._log_timestamp("Evaluating Logistic Regression on test set")
            test_metrics, y_pred, y_proba = self._evaluate_model(model, X_test, y_test, threshold, "test")

            # Log metrics
            mlflow.log_metrics({
                "threshold": threshold,
                "val_f1": val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
            })

            # Save model
            model_path = self.models_dir / "logistic_regression_late_delivery.joblib"
            joblib.dump({"model": model, "threshold": threshold}, model_path)
            mlflow.log_artifact(str(model_path))
            self._log_timestamp(f"Model saved to {model_path}")

            # Print detailed results
            self._print_classification_report(y_test, y_pred, "Logistic Regression")
            self._print_confusion_matrix(y_test, y_pred, "Logistic Regression")
            self._print_feature_importances(model, X_train.columns.tolist(), "Logistic Regression", top_n=15)

            # Store results for comparison
            self.results.append({
                "model": "Logistic Regression",
                "threshold": threshold,
                **test_metrics,
                "y_test": y_test,
                "y_proba": y_proba,
            })

            return model, test_metrics, threshold

    def _optimize_threshold_optuna(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_trials: int = 50,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize decision threshold using Optuna to maximize F1.

        Falls back to PR-curve method if Optuna unavailable.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                thr = trial.suggest_float("threshold", 0.1, 0.9)
                y_pred = (y_proba >= thr).astype(int)
                return f1_score(y_true, y_pred, zero_division=0)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best_threshold = study.best_params["threshold"]
            y_pred = (y_proba >= best_threshold).astype(int)

            metrics = {
                "threshold": best_threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }

            self.logger.info(
                "optuna_threshold_optimized",
                threshold=f"{best_threshold:.4f}",
                f1=f"{metrics['f1']:.4f}",
                trials=n_trials,
            )
            return best_threshold, metrics
        except ImportError:
            self.logger.warning("optuna_not_available, falling back to PR-curve")
            return self._optimize_threshold(y_true, y_proba)

    def train_two_stage(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        dur_train: pd.Series,
        dur_val: pd.Series,
        dur_test: pd.Series,
        sched_train: pd.Series,
        sched_val: pd.Series,
        sched_test: pd.Series,
    ) -> Tuple:
        """
        Train two-stage late delivery predictor.

        Stage 1: XGBRegressor predicts shipping duration.
        Stage 2: XGBClassifier on original features + pred_duration_delta.
        """
        self._log_timestamp("Training Two-Stage Model (Regression + Classification)")

        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count

        with mlflow.start_run(run_name="two_stage"):
            mlflow.log_params({
                "model_type": "two_stage",
                "stage1": "xgboost_regressor",
                "stage2": "xgboost_classifier",
            })

            # Stage 1: Duration regression
            self._log_timestamp("Stage 1: Training duration regressor")
            stage1 = XGBRegressor(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=7,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="rmse",
                verbosity=0,
            )
            try:
                from xgboost.callback import EarlyStopping
                stage1.fit(
                    X_train, dur_train,
                    eval_set=[(X_val, dur_val)],
                    verbose=False,
                    callbacks=[EarlyStopping(rounds=30)],
                )
            except (ImportError, TypeError):
                stage1.fit(X_train, dur_train, eval_set=[(X_val, dur_val)], verbose=False)

            # Build augmented features with predicted duration delta
            pred_dur_train = stage1.predict(X_train)
            pred_dur_val = stage1.predict(X_val)
            pred_dur_test = stage1.predict(X_test)

            X_train_aug = X_train.copy()
            sched_train_arr = np.asarray(sched_train)
            sched_val_arr = np.asarray(sched_val)
            sched_test_arr = np.asarray(sched_test)
            X_train_aug["pred_duration_delta"] = pred_dur_train - sched_train_arr
            X_val_aug = X_val.copy()
            X_val_aug["pred_duration_delta"] = pred_dur_val - sched_val_arr
            X_test_aug = X_test.copy()
            X_test_aug["pred_duration_delta"] = pred_dur_test - sched_test_arr

            # Stage 2: Classification on augmented features
            self._log_timestamp("Stage 2: Training classifier on augmented features")
            stage2 = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
            try:
                from xgboost.callback import EarlyStopping
                stage2.fit(
                    X_train_aug, y_train,
                    eval_set=[(X_val_aug, y_val)],
                    verbose=False,
                    callbacks=[EarlyStopping(rounds=30)],
                )
            except (ImportError, TypeError):
                stage2.fit(X_train_aug, y_train, eval_set=[(X_val_aug, y_val)], verbose=False)

            # Threshold optimization
            y_val_proba = stage2.predict_proba(X_val_aug)[:, 1]
            threshold, val_metrics = self._optimize_threshold_optuna(y_val, y_val_proba)

            # Evaluate on test
            self._log_timestamp("Evaluating two-stage model on test set")
            test_metrics, y_pred, y_proba = self._evaluate_model(
                stage2, X_test_aug, y_test, threshold, "test"
            )

            # Log metrics
            mlflow.log_metrics({
                "threshold": threshold,
                "val_f1": val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
            })

            # Save both stages
            model_path = self.models_dir / "two_stage_late_delivery.joblib"
            joblib.dump({
                "stage1_regressor": stage1,
                "stage2_classifier": stage2,
                "threshold": threshold,
                "features": X_train.columns.tolist(),
                "model_name": "two_stage_late_delivery",
                "model_version": "1.0",
            }, model_path)
            mlflow.log_artifact(str(model_path))
            self._log_timestamp(f"Two-stage model saved to {model_path}")

            # Print results
            self._print_classification_report(y_test, y_pred, "Two-Stage (Reg+Cls)")
            self._print_confusion_matrix(y_test, y_pred, "Two-Stage (Reg+Cls)")
            self._print_feature_importances(stage2, X_train_aug.columns.tolist(), "Two-Stage (Stage2)", top_n=15)

            self.results.append({
                "model": "Two-Stage",
                "threshold": threshold,
                **test_metrics,
                "y_test": y_test,
                "y_proba": y_proba,
            })

            return stage1, stage2, test_metrics, threshold

    def train_stacked_ensemble(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> Tuple:
        """
        Train stacked ensemble: XGBoost + RF + LR base learners with
        LogisticRegression meta-learner using out-of-fold predictions.
        """
        from sklearn.model_selection import KFold

        self._log_timestamp("Training Stacked Ensemble (XGB + RF + LR -> Meta-LR)")

        with mlflow.start_run(run_name="stacked_ensemble"):
            mlflow.log_params({
                "model_type": "stacked_ensemble",
                "base_models": "xgboost,random_forest,logistic_regression",
                "meta_learner": "logistic_regression",
                "n_folds": 5,
            })

            # Load pre-trained base models
            base_model_paths = {
                "xgb": self.models_dir / "xgboost_late_delivery.joblib",
                "rf": self.models_dir / "random_forest_late_delivery.joblib",
                "lr": self.models_dir / "logistic_regression_late_delivery.joblib",
            }

            base_models = {}
            for name, path in base_model_paths.items():
                if path.exists():
                    artifact = joblib.load(path)
                    base_models[name] = artifact["model"]
                else:
                    self.logger.warning(f"base_model_missing", model=name, path=str(path))

            if len(base_models) < 2:
                self.logger.error("insufficient_base_models", count=len(base_models))
                return None, {}, 0.5

            # Build OOF meta-features via 5-fold CV
            self._log_timestamp("Building out-of-fold meta-features")
            kf = KFold(n_splits=5, shuffle=False)  # No shuffle for temporal data
            oof_preds = np.zeros((len(X_train), len(base_models)))

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]

                for m_idx, (m_name, m_template) in enumerate(base_models.items()):
                    # Clone and retrain on fold
                    if isinstance(m_template, Pipeline):
                        m_clone = Pipeline([
                            (step_name, type(step)(**step.get_params()))
                            for step_name, step in m_template.steps
                        ])
                    else:
                        m_clone = type(m_template)(**m_template.get_params())
                    m_clone.fit(X_fold_train, y_fold_train)
                    oof_preds[val_idx, m_idx] = m_clone.predict_proba(X_fold_val)[:, 1]

            # Train meta-learner on OOF predictions
            self._log_timestamp("Training meta-learner on OOF predictions")
            meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            meta_learner.fit(oof_preds, y_train)

            # Build val/test meta-features from full base models
            val_meta = np.column_stack([
                m.predict_proba(X_val)[:, 1] for m in base_models.values()
            ])
            test_meta = np.column_stack([
                m.predict_proba(X_test)[:, 1] for m in base_models.values()
            ])

            # Threshold optimization
            val_proba = meta_learner.predict_proba(val_meta)[:, 1]
            threshold, val_metrics = self._optimize_threshold_optuna(y_val, val_proba)

            # Evaluate on test
            self._log_timestamp("Evaluating stacked ensemble on test set")
            test_proba = meta_learner.predict_proba(test_meta)[:, 1]
            y_pred = (test_proba >= threshold).astype(int)

            test_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, test_proba),
            }
            pr, rc, _ = precision_recall_curve(y_test, test_proba)
            test_metrics["pr_auc"] = auc(rc, pr)

            self.logger.info(
                "stacked_ensemble_evaluated",
                test_f1=f"{test_metrics['f1']:.4f}",
                test_roc_auc=f"{test_metrics['roc_auc']:.4f}",
            )

            # Log metrics
            mlflow.log_metrics({
                "threshold": threshold,
                "val_f1": val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
            })

            # Save ensemble
            model_path = self.models_dir / "stacked_ensemble_late_delivery.joblib"
            joblib.dump({
                "base_models": base_models,
                "meta_learner": meta_learner,
                "threshold": threshold,
                "base_model_names": list(base_models.keys()),
                "model_name": "stacked_ensemble_late_delivery",
                "model_version": "1.0",
            }, model_path)
            mlflow.log_artifact(str(model_path))
            self._log_timestamp(f"Stacked ensemble saved to {model_path}")

            # Print results
            self._print_classification_report(y_test, y_pred, "Stacked Ensemble")
            self._print_confusion_matrix(y_test, y_pred, "Stacked Ensemble")

            self.results.append({
                "model": "Stacked Ensemble",
                "threshold": threshold,
                **test_metrics,
                "y_test": y_test,
                "y_proba": test_proba,
            })

            return meta_learner, test_metrics, threshold

    def print_comparison_table(self):
        """Print comparison table of all trained models."""
        if not self.results:
            return

        print(f"\n{'='*100}")
        print("MODEL COMPARISON - LATE DELIVERY PREDICTION")
        print(f"{'='*100}")
        print(f"{'Model':<20} {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'PR-AUC':>10}")
        print(f"{'-'*100}")

        for result in self.results:
            print(
                f"{result['model']:<20} "
                f"{result['threshold']:>10.4f} "
                f"{result['accuracy']:>10.4f} "
                f"{result['precision']:>10.4f} "
                f"{result['recall']:>10.4f} "
                f"{result['f1']:>10.4f} "
                f"{result['roc_auc']:>10.4f} "
                f"{result['pr_auc']:>10.4f}"
            )

        print(f"{'='*100}\n")

        # Identify best model for each metric
        best_f1_idx = np.argmax([r["f1"] for r in self.results])
        best_auc_idx = np.argmax([r["roc_auc"] for r in self.results])

        print(f"Best F1 Score:  {self.results[best_f1_idx]['model']} ({self.results[best_f1_idx]['f1']:.4f})")
        print(f"Best ROC-AUC:   {self.results[best_auc_idx]['model']} ({self.results[best_auc_idx]['roc_auc']:.4f})")
        print()

    def plot_roc_comparison(self, output_path: str = "reports/delivery_roc_comparison.png"):
        """
        Plot ROC curves for all models.

        Args:
            output_path: Path to save the plot
        """
        if not self.results:
            return

        self._log_timestamp(f"Generating ROC curve comparison plot")

        plt.figure(figsize=(10, 8))

        for result in self.results:
            y_test = result["y_test"]
            y_proba = result["y_proba"]

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = result["roc_auc"]

            plt.plot(fpr, tpr, linewidth=2, label=f"{result['model']} (AUC = {roc_auc:.4f})")

        # Plot diagonal
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve Comparison - Late Delivery Prediction", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        self._log_timestamp(f"ROC comparison plot saved to {output_path}")

        # Log to MLflow
        mlflow.log_artifact(str(output_path))

        plt.close()


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train late delivery prediction models on Olist e-commerce data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["xgboost", "random_forest", "logistic", "two_stage", "ensemble", "all"],
        help="Model to train (default: all)",
    )
    args = parser.parse_args()

    print(f"\n{'='*100}")
    print("LATE DELIVERY PREDICTION MODEL TRAINING")
    print(f"{'='*100}\n")

    # Initialize data loader
    print("[INFO] Initializing data loader")
    loader = OlistDataLoader()

    # Load and prepare data
    print("[INFO] Loading and preparing late delivery dataset")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_late_delivery_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )

    print(f"\n[INFO] Dataset summary:")
    print(f"  Training set:   {len(X_train):,} samples ({y_train.mean():.2%} late)")
    print(f"  Validation set: {len(X_val):,} samples ({y_val.mean():.2%} late)")
    print(f"  Test set:       {len(X_test):,} samples ({y_test.mean():.2%} late)")
    print(f"  Features:       {X_train.shape[1]}")
    print()

    # Initialize trainer
    trainer = LateDeliveryTrainer()

    # Train models
    if args.model in ["xgboost", "all"]:
        trainer.train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)

    if args.model in ["random_forest", "all"]:
        trainer.train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)

    if args.model in ["logistic", "all"]:
        trainer.train_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test)

    if args.model in ["two_stage", "all"]:
        # Load duration data for two-stage model
        try:
            result = loader.prepare_late_delivery_data_with_duration(
                test_size=0.2, val_size=0.1, random_state=42,
            )
            (X_train_d, X_val_d, X_test_d, y_train_d, y_val_d, y_test_d,
             dur_train, dur_val, dur_test, sched_train, sched_val, sched_test) = result
            trainer.train_two_stage(
                X_train_d, X_val_d, X_test_d, y_train_d, y_val_d, y_test_d,
                dur_train, dur_val, dur_test, sched_train, sched_val, sched_test,
            )
        except Exception as e:
            print(f"[WARNING] Two-stage training failed: {e}")
            logger.warning("two_stage_training_failed", error=str(e))

    if args.model in ["ensemble", "all"]:
        # Stacked ensemble requires base models to be trained first
        trainer.train_stacked_ensemble(X_train, X_val, X_test, y_train, y_val, y_test)

    # Print comparison table
    trainer.print_comparison_table()

    # Generate ROC comparison plot
    if len(trainer.results) > 1:
        trainer.plot_roc_comparison()

    print(f"\n{'='*100}")
    print("TRAINING COMPLETE")
    print(f"{'='*100}")
    print(f"Models saved to: {trainer.models_dir}")
    print(f"MLflow experiment: {trainer.mlflow_experiment}")
    print(f"  View results: mlflow ui")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
