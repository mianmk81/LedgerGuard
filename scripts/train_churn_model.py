#!/usr/bin/env python3
"""
Churn Prediction Model Training Script for LedgerGuard.

Trains and evaluates multiple machine learning models for customer churn prediction
using Olist e-commerce customer data. Implements MLflow tracking, hyperparameter
optimization, and comprehensive performance evaluation.

Models trained:
1. LightGBM Classifier (gradient boosting)
2. Logistic Regression (baseline linear model)
3. Random Forest Classifier (ensemble method)

Usage:
    python scripts/train_churn_model.py --model all
    python scripts/train_churn_model.py --model lgbm
    python scripts/train_churn_model.py --model logistic
    python scripts/train_churn_model.py --model random_forest
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_loader import OlistDataLoader


class ChurnModelTrainer:
    """
    Orchestrates training, evaluation, and tracking of churn prediction models.

    Implements threshold tuning on validation set, comprehensive metrics tracking
    via MLflow, and model persistence for deployment.
    """

    def __init__(
        self,
        experiment_name: str = "ledgerguard-churn-prediction",
        models_dir: str = "models/churn",
        reports_dir: str = "reports",
    ):
        """
        Initialize churn model trainer.

        Args:
            experiment_name: MLflow experiment name for tracking
            models_dir: Directory to save trained model artifacts
            reports_dir: Directory to save evaluation reports and plots
        """
        self.experiment_name = experiment_name
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)

        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow
        mlflow.set_experiment(experiment_name)

        # Storage for model results
        self.results: List[Dict[str, Any]] = []

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

    def log_with_timestamp(self, message: str) -> None:
        """Print message with timestamp for progress tracking."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold using precision-recall curve.

        Optimizes for F1 score on validation set to balance precision and recall.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class

        Returns:
            Tuple of (optimal_threshold, best_f1_score)
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # Find threshold that maximizes F1
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        return optimal_threshold, best_f1

    def evaluate_model(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_name: str,
    ) -> Tuple[Dict[str, float], float, np.ndarray, np.ndarray]:
        """
        Comprehensive model evaluation with threshold tuning.

        Args:
            model: Trained model with predict_proba method
            X_val: Validation features for threshold tuning
            y_val: Validation labels
            X_test: Test features for final evaluation
            y_test: Test labels
            model_name: Name of the model for logging

        Returns:
            Tuple of (metrics_dict, optimal_threshold, y_test_pred, y_test_proba)
        """
        self.log_with_timestamp(f"Evaluating {model_name}...")

        # Get validation predictions
        y_val_proba = model.predict_proba(X_val)[:, 1]

        # Find optimal threshold on validation set
        optimal_threshold, val_f1 = self.find_optimal_threshold(y_val, y_val_proba)
        self.log_with_timestamp(
            f"  Optimal threshold: {optimal_threshold:.4f} (Val F1: {val_f1:.4f})"
        )

        # Get test predictions with optimal threshold
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

        # Calculate comprehensive metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_test_proba),
            "auc_pr": average_precision_score(y_test, y_test_proba),
            "optimal_threshold": optimal_threshold,
        }

        self.log_with_timestamp(f"  Test Metrics:")
        self.log_with_timestamp(f"    Accuracy:  {metrics['accuracy']:.4f}")
        self.log_with_timestamp(f"    Precision: {metrics['precision']:.4f}")
        self.log_with_timestamp(f"    Recall:    {metrics['recall']:.4f}")
        self.log_with_timestamp(f"    F1 Score:  {metrics['f1']:.4f}")
        self.log_with_timestamp(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
        self.log_with_timestamp(f"    AUC-PR:    {metrics['auc_pr']:.4f}")

        return metrics, optimal_threshold, y_test_pred, y_test_proba

    def print_evaluation_details(
        self,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        model_name: str,
        feature_names: List[str] = None,
        feature_importances: np.ndarray = None,
    ) -> None:
        """
        Print detailed evaluation including classification report and confusion matrix.

        Args:
            y_test: True test labels
            y_test_pred: Predicted test labels
            model_name: Name of the model
            feature_names: List of feature names (optional)
            feature_importances: Feature importance scores (optional)
        """
        self.log_with_timestamp(f"\n{'='*80}")
        self.log_with_timestamp(f"{model_name} - Detailed Evaluation")
        self.log_with_timestamp(f"{'='*80}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=["Retained", "Churned"]))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Retained  Churned")
        print(f"Actual Retained {cm[0, 0]:8d}  {cm[0, 1]:7d}")
        print(f"       Churned  {cm[1, 0]:8d}  {cm[1, 1]:7d}")

        # Feature importances (top 15)
        if feature_importances is not None and feature_names is not None:
            print("\nTop 15 Most Important Features:")
            # Sort by importance
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": feature_importances}
            ).sort_values("importance", ascending=False)

            for idx, row in importance_df.head(15).iterrows():
                print(f"  {row['feature']:40s} {row['importance']:.6f}")

    def plot_feature_importance(
        self,
        feature_names: List[str],
        feature_importances: np.ndarray,
        model_name: str,
    ) -> str:
        """
        Create and save feature importance plot.

        Args:
            feature_names: List of feature names
            feature_importances: Importance scores
            model_name: Name of the model

        Returns:
            Path to saved plot
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importances}
        ).sort_values("importance", ascending=False)

        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)

        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importance Score")
        plt.title(f"{model_name} - Top 20 Feature Importances")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        plot_path = self.reports_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.log_with_timestamp(f"  Feature importance plot saved to: {plot_path}")
        return str(plot_path)

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> None:
        """Train and evaluate LightGBM classifier."""
        model_name = "LightGBM Classifier"
        self.log_with_timestamp(f"\n{'='*80}")
        self.log_with_timestamp(f"Training {model_name}")
        self.log_with_timestamp(f"{'='*80}")

        with mlflow.start_run(run_name="lightgbm_churn"):
            # Compute scale_pos_weight for imbalanced churn (churn=1 is positive)
            churn_rate = float(y_train.mean())
            scale_pos_weight = (1 - churn_rate) / (churn_rate + 1e-10) if churn_rate < 0.99 else 0.1

            # Define hyperparameters
            params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "class_weight": "balanced",
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
                "verbose": -1,
            }

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("train_churn_rate", churn_rate)

            # Train model
            self.log_with_timestamp("Training model...")
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

            # Evaluate
            metrics, threshold, y_pred, y_proba = self.evaluate_model(
                model, X_val, y_val, X_test, y_test, model_name
            )

            # Log metrics
            mlflow.log_metrics(metrics)

            # Get feature importances
            feature_importances = model.feature_importances_
            feature_names = X_train.columns.tolist()

            # Plot and log feature importance
            plot_path = self.plot_feature_importance(
                feature_names, feature_importances, model_name
            )
            mlflow.log_artifact(plot_path)

            # Print detailed evaluation
            self.print_evaluation_details(
                y_test, y_pred, model_name, feature_names, feature_importances
            )

            # Save model
            model_path = self.models_dir / "lightgbm_churn_model.pkl"
            joblib.dump(
                {"model": model, "threshold": threshold, "feature_names": feature_names},
                model_path,
            )
            mlflow.log_artifact(str(model_path))
            self.log_with_timestamp(f"Model saved to: {model_path}")

            # Store results
            self.results.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc_roc": metrics["auc_roc"],
                    "auc_pr": metrics["auc_pr"],
                    "threshold": threshold,
                    "y_proba": y_proba,
                }
            )

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> None:
        """Train and evaluate Logistic Regression with StandardScaler."""
        model_name = "Logistic Regression"
        self.log_with_timestamp(f"\n{'='*80}")
        self.log_with_timestamp(f"Training {model_name}")
        self.log_with_timestamp(f"{'='*80}")

        with mlflow.start_run(run_name="logistic_regression_churn"):
            # Define hyperparameters
            params = {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "solver": "lbfgs",
            }

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "logistic_regression")
            mlflow.log_param("scaling", "standard_scaler")

            # Create pipeline with scaling
            self.log_with_timestamp("Training model with StandardScaler...")
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(**params)),
                ]
            )

            # Train model
            pipeline.fit(X_train, y_train)

            # Evaluate
            metrics, threshold, y_pred, y_proba = self.evaluate_model(
                pipeline, X_val, y_val, X_test, y_test, model_name
            )

            # Log metrics
            mlflow.log_metrics(metrics)

            # Get feature importances (coefficients)
            feature_importances = np.abs(pipeline.named_steps["classifier"].coef_[0])
            feature_names = X_train.columns.tolist()

            # Plot and log feature importance
            plot_path = self.plot_feature_importance(
                feature_names, feature_importances, model_name
            )
            mlflow.log_artifact(plot_path)

            # Print detailed evaluation
            self.print_evaluation_details(
                y_test, y_pred, model_name, feature_names, feature_importances
            )

            # Save model
            model_path = self.models_dir / "logistic_regression_churn_model.pkl"
            joblib.dump(
                {"model": pipeline, "threshold": threshold, "feature_names": feature_names},
                model_path,
            )
            mlflow.log_artifact(str(model_path))
            self.log_with_timestamp(f"Model saved to: {model_path}")

            # Store results
            self.results.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc_roc": metrics["auc_roc"],
                    "auc_pr": metrics["auc_pr"],
                    "threshold": threshold,
                    "y_proba": y_proba,
                }
            )

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> None:
        """Train and evaluate Random Forest classifier."""
        model_name = "Random Forest"
        self.log_with_timestamp(f"\n{'='*80}")
        self.log_with_timestamp(f"Training {model_name}")
        self.log_with_timestamp(f"{'='*80}")

        with mlflow.start_run(run_name="random_forest_churn"):
            # Define hyperparameters
            params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "random_forest")

            # Train model
            self.log_with_timestamp("Training model...")
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # Evaluate
            metrics, threshold, y_pred, y_proba = self.evaluate_model(
                model, X_val, y_val, X_test, y_test, model_name
            )

            # Log metrics
            mlflow.log_metrics(metrics)

            # Get feature importances
            feature_importances = model.feature_importances_
            feature_names = X_train.columns.tolist()

            # Plot and log feature importance
            plot_path = self.plot_feature_importance(
                feature_names, feature_importances, model_name
            )
            mlflow.log_artifact(plot_path)

            # Print detailed evaluation
            self.print_evaluation_details(
                y_test, y_pred, model_name, feature_names, feature_importances
            )

            # Save model
            model_path = self.models_dir / "random_forest_churn_model.pkl"
            joblib.dump(
                {"model": model, "threshold": threshold, "feature_names": feature_names},
                model_path,
            )
            mlflow.log_artifact(str(model_path))
            self.log_with_timestamp(f"Model saved to: {model_path}")

            # Store results
            self.results.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc_roc": metrics["auc_roc"],
                    "auc_pr": metrics["auc_pr"],
                    "threshold": threshold,
                    "y_proba": y_proba,
                }
            )

    def plot_roc_comparison(self, y_test: np.ndarray) -> None:
        """
        Create ROC curve comparison plot for all trained models.

        Args:
            y_test: True test labels
        """
        if not self.results:
            self.log_with_timestamp("No models to compare")
            return

        self.log_with_timestamp("\nGenerating ROC comparison plot...")

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each model
        for result in self.results:
            fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
            auc_score = result["auc_roc"]
            plt.plot(
                fpr,
                tpr,
                label=f"{result['model']} (AUC = {auc_score:.4f})",
                linewidth=2,
            )

        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)

        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve Comparison - Churn Prediction Models", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = self.reports_dir / "churn_roc_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.log_with_timestamp(f"ROC comparison plot saved to: {plot_path}")

    def print_comparison_table(self) -> None:
        """Print comparison table of all trained models."""
        if not self.results:
            self.log_with_timestamp("No models to compare")
            return

        self.log_with_timestamp(f"\n{'='*80}")
        self.log_with_timestamp("MODEL COMPARISON SUMMARY")
        self.log_with_timestamp(f"{'='*80}\n")

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results)[
            ["model", "accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "threshold"]
        ]

        # Format for display
        print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Find best model by F1 score
        best_idx = comparison_df["f1"].idxmax()
        best_model = comparison_df.loc[best_idx, "model"]
        best_f1 = comparison_df.loc[best_idx, "f1"]

        self.log_with_timestamp(f"\nBest Model: {best_model} (F1 Score: {best_f1:.4f})")

        # Save comparison to CSV
        csv_path = self.reports_dir / "churn_model_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        self.log_with_timestamp(f"Comparison table saved to: {csv_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Train churn prediction models for LedgerGuard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_churn_model.py --model all
  python scripts/train_churn_model.py --model lgbm
  python scripts/train_churn_model.py --model logistic
  python scripts/train_churn_model.py --model random_forest
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgbm", "logistic", "random_forest", "all"],
        default="all",
        help="Model to train (default: all)",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = ChurnModelTrainer()

    trainer.log_with_timestamp("="*80)
    trainer.log_with_timestamp("LEDGERGUARD CHURN PREDICTION MODEL TRAINING")
    trainer.log_with_timestamp("="*80)

    # Load data
    trainer.log_with_timestamp("\nLoading Olist customer data...")
    try:
        loader = OlistDataLoader()
        X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_churn_data()

        trainer.log_with_timestamp(f"Training set:   {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        trainer.log_with_timestamp(f"Validation set: {X_val.shape[0]:,} samples")
        trainer.log_with_timestamp(f"Test set:       {X_test.shape[0]:,} samples")
        trainer.log_with_timestamp(f"Churn rate (train): {y_train.mean():.2%}")
        trainer.log_with_timestamp(f"Churn rate (val):   {y_val.mean():.2%}")
        trainer.log_with_timestamp(f"Churn rate (test):  {y_test.mean():.2%}")

    except Exception as e:
        trainer.log_with_timestamp(f"ERROR: Failed to load data: {e}")
        sys.exit(1)

    # Train selected models
    try:
        if args.model in ["lgbm", "all"]:
            trainer.train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)

        if args.model in ["logistic", "all"]:
            trainer.train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)

        if args.model in ["random_forest", "all"]:
            trainer.train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)

    except Exception as e:
        trainer.log_with_timestamp(f"ERROR: Model training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate comparison artifacts
    if len(trainer.results) > 1:
        trainer.plot_roc_comparison(y_test)

    trainer.print_comparison_table()

    trainer.log_with_timestamp("\n" + "="*80)
    trainer.log_with_timestamp("TRAINING COMPLETE")
    trainer.log_with_timestamp("="*80)
    trainer.log_with_timestamp(f"\nModels saved to: {trainer.models_dir.absolute()}")
    trainer.log_with_timestamp(f"Reports saved to: {trainer.reports_dir.absolute()}")
    trainer.log_with_timestamp(f"MLflow tracking: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
