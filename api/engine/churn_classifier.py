"""
Churn Classifier Module

LightGBM-based supervised classifier for customer churn prediction.
Trained on Telco Churn dataset with proper time-based splits,
hyperparameter tuning via Optuna, and calibration via Platt scaling.

Integrated with MLflow for experiment tracking and model registry.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


class ChurnClassifier:
    """
    LightGBM churn classifier with MLflow tracking.

    This classifier predicts customer churn probability using a LightGBM
    gradient boosting model with optional Optuna hyperparameter tuning
    and Platt scaling calibration. All experiments are tracked in MLflow
    for reproducibility and model versioning.

    Features:
    - Automatic preprocessing of Telco Churn dataset
    - Time-based train/validation/test splits (70/15/15)
    - Baseline logistic regression for comparison
    - Optuna hyperparameter optimization
    - Platt scaling for probability calibration
    - MLflow experiment tracking and model registry
    - Feature importance analysis

    Attributes:
        model: Trained LightGBM classifier
        calibrator: CalibratedClassifierCV wrapper for probability calibration
        feature_names: List of feature names used during training
        label_encoders: Dictionary of LabelEncoder instances for categorical features
        experiment_name: MLflow experiment name
        logger: Structlog logger instance
    """

    def __init__(self, experiment_name: str = "bre_churn_classifier"):
        """
        Initialize the churn classifier.

        Args:
            experiment_name: MLflow experiment name for tracking runs
        """
        self.model: Optional[lgb.LGBMClassifier] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.feature_names: Optional[list[str]] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.experiment_name = experiment_name
        self.logger = structlog.get_logger()

        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the Telco Churn dataset.

        Handles:
        - TotalCharges whitespace to NaN conversion
        - Categorical encoding via LabelEncoder
        - Feature engineering (tenure groups, derived ratios)

        Args:
            df: Raw DataFrame with Telco Churn schema

        Returns:
            Preprocessed DataFrame with engineered features
        """
        df = df.copy()

        # Handle TotalCharges: convert whitespace to NaN, then to float
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            )
            # Fill NaN with 0 (typically new customers with no charges yet)
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        # Feature engineering
        if "tenure" in df.columns:
            # Tenure groups: 0-12, 13-24, 25-48, 49+
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 100],
                labels=["0-12", "13-24", "25-48", "49+"],
            )

        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            # Average charges per month of tenure
            df["charges_per_month_tenure"] = df["MonthlyCharges"] / (
                df["tenure"] + 1
            )  # +1 to avoid division by zero

        # Premium services indicator
        premium_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        available_premium = [col for col in premium_cols if col in df.columns]
        if available_premium:
            df["has_premium_services"] = (
                df[available_premium]
                .apply(lambda row: (row == "Yes").sum(), axis=1)
                .astype(int)
            )

        # Identify categorical columns (object dtype)
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Exclude target column if present
        if "Churn" in categorical_cols:
            categorical_cols.remove("Churn")

        # Label encode categorical features
        for col in categorical_cols:
            if col not in self.label_encoders:
                # Training mode: fit new encoder
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(
                    df[col].astype(str)
                )
            else:
                # Prediction mode: use existing encoder
                try:
                    df[col] = self.label_encoders[col].transform(
                        df[col].astype(str)
                    )
                except ValueError:
                    # Handle unseen labels by mapping to most frequent class
                    self.logger.warning(
                        "unseen_category_in_prediction",
                        column=col,
                        action="mapping_to_mode",
                    )
                    # Map unseen to -1, then replace with mode
                    df[col] = df[col].map(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_
                        else -1
                    )
                    mode_value = 0  # Default to first class
                    df[col] = df[col].replace(-1, mode_value)

        return df

    def _compute_dataset_hash(self, df: pd.DataFrame) -> str:
        """
        Compute a hash of the training dataset for versioning.

        Args:
            df: Training DataFrame

        Returns:
            SHA256 hash of dataset shape and first/last rows
        """
        # Use shape + sample rows to create reproducible hash
        signature = f"{df.shape}_{df.head(1).to_json()}_{df.tail(1).to_json()}"
        return hashlib.sha256(signature.encode()).hexdigest()[:16]

    def _create_splits(
        self, df: pd.DataFrame, target_col: str = "Churn"
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Create time-based train/validation/test splits.

        Uses tenure as a proxy for time to ensure temporal ordering.
        Split ratios: 70% train, 15% validation, 15% test.

        Args:
            df: Preprocessed DataFrame
            target_col: Name of target column

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Sort by tenure (proxy for customer age/time)
        df_sorted = df.sort_values("tenure").reset_index(drop=True)

        # Compute split indices
        n = len(df_sorted)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        # Split data
        train = df_sorted.iloc[:train_end]
        val = df_sorted.iloc[train_end:val_end]
        test = df_sorted.iloc[val_end:]

        # Separate features and target
        y_train = train[target_col].map({"Yes": 1, "No": 0})
        y_val = val[target_col].map({"Yes": 1, "No": 0})
        y_test = test[target_col].map({"Yes": 1, "No": 0})

        X_train = train.drop(columns=[target_col])
        X_val = val.drop(columns=[target_col])
        X_test = test.drop(columns=[target_col])

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        self.logger.info(
            "created_time_based_splits",
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test),
            train_churn_rate=y_train.mean(),
            val_churn_rate=y_val.mean(),
            test_churn_rate=y_test.mean(),
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _train_baseline(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """
        Train a baseline logistic regression model for comparison.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of baseline metrics
        """
        baseline = LogisticRegression(max_iter=1000, random_state=42)
        baseline.fit(X_train, y_train)

        y_pred = baseline.predict(X_test)
        y_proba = baseline.predict_proba(X_test)[:, 1]

        metrics = {
            "baseline_auroc": roc_auc_score(y_test, y_proba),
            "baseline_f1": f1_score(y_test, y_pred),
            "baseline_precision": precision_score(y_test, y_pred),
            "baseline_recall": recall_score(y_test, y_pred),
            "baseline_accuracy": accuracy_score(y_test, y_pred),
        }

        self.logger.info("baseline_model_trained", **metrics)

        return metrics

    def _optuna_objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """
        Optuna optimization objective function.

        Args:
            trial: Optuna trial instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Validation AUROC score (to maximize)
        """
        # Define hyperparameter search space
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": 42,
        }

        # Train model
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        # Evaluate on validation set
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, y_val_proba)

        return auroc

    def _tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of Optuna trials

        Returns:
            Best hyperparameters dictionary
        """
        self.logger.info("starting_hyperparameter_tuning", n_trials=n_trials)

        study = optuna.create_study(
            direction="maximize",
            study_name="churn_classifier_hpo",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            lambda trial: self._optuna_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_params.update(
            {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "random_state": 42,
            }
        )

        self.logger.info(
            "hyperparameter_tuning_complete",
            best_auroc=study.best_value,
            best_params=best_params,
        )

        return best_params

    def train(
        self, df: pd.DataFrame, target_col: str = "Churn", tune: bool = True, n_trials: int = 50
    ) -> dict[str, Any]:
        """
        Train the churn classifier.

        Complete training pipeline:
        1. Preprocess data and engineer features
        2. Create time-based splits (70/15/15)
        3. Train baseline logistic regression
        4. Train LightGBM (with optional hyperparameter tuning)
        5. Calibrate probabilities via Platt scaling
        6. Evaluate on test set
        7. Log all artifacts to MLflow

        Args:
            df: DataFrame with features and target (Telco Churn schema)
            target_col: Name of binary target column (default "Churn")
            tune: Whether to use Optuna hyperparameter tuning
            n_trials: Number of Optuna trials (if tune=True)

        Returns:
            Dictionary with metrics: auroc, f1, precision, recall, accuracy on test set
        """
        with mlflow.start_run(run_name=f"churn_classifier_tune={tune}"):
            self.logger.info(
                "training_started",
                dataset_shape=df.shape,
                target_col=target_col,
                tune=tune,
            )

            # Compute dataset hash for versioning
            dataset_hash = self._compute_dataset_hash(df)
            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_param("dataset_rows", len(df))
            mlflow.log_param("dataset_cols", len(df.columns))

            # Preprocess data
            df_processed = self._preprocess_data(df)

            # Create splits
            X_train, y_train, X_val, y_val, X_test, y_test = self._create_splits(
                df_processed, target_col
            )

            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", len(self.feature_names))

            # Train baseline model
            baseline_metrics = self._train_baseline(X_train, y_train, X_test, y_test)
            for key, value in baseline_metrics.items():
                mlflow.log_metric(key, value)

            # Determine hyperparameters
            if tune:
                best_params = self._tune_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=n_trials
                )
                mlflow.log_param("tuning_method", "optuna")
                mlflow.log_param("n_trials", n_trials)
            else:
                # Default hyperparameters
                best_params = {
                    "objective": "binary",
                    "metric": "auc",
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "n_estimators": 200,
                    "random_state": 42,
                }
                mlflow.log_param("tuning_method", "default")

            # Log hyperparameters
            for param, value in best_params.items():
                mlflow.log_param(f"lgbm_{param}", value)

            # Train final model
            self.logger.info("training_lightgbm_model", params=best_params)
            self.model = lgb.LGBMClassifier(**best_params)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

            # Calibrate probabilities using validation set
            self.logger.info("calibrating_probabilities", method="platt_scaling")
            self.calibrator = CalibratedClassifierCV(
                self.model, method="sigmoid", cv="prefit"
            )
            self.calibrator.fit(X_val, y_val)

            # Evaluate on test set
            y_test_pred = self.calibrator.predict(X_test)
            y_test_proba = self.calibrator.predict_proba(X_test)[:, 1]

            metrics = {
                "test_auroc": roc_auc_score(y_test, y_test_proba),
                "test_f1": f1_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred),
                "test_recall": recall_score(y_test, y_test_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
            }

            # Log test metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            mlflow.log_metric("test_true_negatives", int(cm[0, 0]))
            mlflow.log_metric("test_false_positives", int(cm[0, 1]))
            mlflow.log_metric("test_false_negatives", int(cm[1, 0]))
            mlflow.log_metric("test_true_positives", int(cm[1, 1]))

            # Feature importance
            feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            self.logger.info(
                "top_features",
                features=feature_importance.head(10).to_dict("records"),
            )

            # Log feature importance as JSON
            mlflow.log_dict(
                feature_importance.to_dict("records"), "feature_importance.json"
            )

            # Log models
            mlflow.lightgbm.log_model(self.model, "lightgbm_model")
            mlflow.sklearn.log_model(self.calibrator, "calibrated_model")

            # Log preprocessing artifacts
            mlflow.log_dict(
                {k: v.classes_.tolist() for k, v in self.label_encoders.items()},
                "label_encoders.json",
            )
            mlflow.log_dict({"features": self.feature_names}, "feature_names.json")

            self.logger.info("training_complete", **metrics)

            return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated churn probabilities.

        Args:
            df: DataFrame with same schema as training data (without target)

        Returns:
            Array of churn probabilities (shape: [n_samples])

        Raises:
            ValueError: If model has not been trained
        """
        if self.calibrator is None or self.model is None:
            raise ValueError(
                "Model has not been trained. Call train() first or load a saved model."
            )

        # Preprocess data
        df_processed = self._preprocess_data(df)

        # Ensure feature alignment
        if self.feature_names is not None:
            # Reorder and select features to match training
            df_processed = df_processed[self.feature_names]

        # Predict probabilities (calibrated)
        probabilities = self.calibrator.predict_proba(df_processed)[:, 1]

        return probabilities

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary churn labels.

        Args:
            df: DataFrame with same schema as training data (without target)
            threshold: Probability threshold for positive class (default 0.5)

        Returns:
            Array of binary predictions (0 or 1, shape: [n_samples])

        Raises:
            ValueError: If model has not been trained
        """
        probabilities = self.predict_proba(df)
        return (probabilities >= threshold).astype(int)

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Saves:
        - LightGBM model (booster)
        - Calibrator (sklearn CalibratedClassifierCV)
        - Label encoders
        - Feature names

        Args:
            path: Directory path to save model artifacts

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None or self.calibrator is None:
            raise ValueError("Model has not been trained. Call train() first.")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM booster
        self.model.booster_.save_model(str(save_path / "lightgbm_model.txt"))

        # Save calibrator using pickle (via mlflow)
        import pickle

        with open(save_path / "calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator, f)

        # Save label encoders
        label_encoder_classes = {
            k: v.classes_.tolist() for k, v in self.label_encoders.items()
        }
        with open(save_path / "label_encoders.json", "w") as f:
            json.dump(label_encoder_classes, f, indent=2)

        # Save feature names
        with open(save_path / "feature_names.json", "w") as f:
            json.dump({"features": self.feature_names}, f, indent=2)

        self.logger.info("model_saved", path=str(save_path))

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Loads:
        - LightGBM model (booster)
        - Calibrator (sklearn CalibratedClassifierCV)
        - Label encoders
        - Feature names

        Args:
            path: Directory path containing model artifacts

        Raises:
            FileNotFoundError: If model artifacts are not found
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")

        # Load LightGBM booster
        self.model = lgb.Booster(model_file=str(load_path / "lightgbm_model.txt"))

        # Load calibrator
        import pickle

        with open(load_path / "calibrator.pkl", "rb") as f:
            self.calibrator = pickle.load(f)

        # Load label encoders
        with open(load_path / "label_encoders.json", "r") as f:
            label_encoder_classes = json.load(f)

        self.label_encoders = {}
        for col, classes in label_encoder_classes.items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            self.label_encoders[col] = le

        # Load feature names
        with open(load_path / "feature_names.json", "r") as f:
            feature_data = json.load(f)
            self.feature_names = feature_data["features"]

        self.logger.info("model_loaded", path=str(load_path))
