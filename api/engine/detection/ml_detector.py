"""
Machine Learning Anomaly Detection — 4-Model Ensemble or Pretrained NAB.

This module implements Layer 2 detection: unsupervised ML-based anomaly detection.

Modes:
    1. Pretrained (use_pretrained=True): Loads NAB Isolation Forest from
       models/anomaly_industry/isolation_forest_nab.joblib. No training at runtime.
       Converts multivariate daily metrics to univariate proxy + 5 rolling features.
    2. Train-at-runtime (use_pretrained=False): 4-model weighted ensemble with
       IF, OCSVM, LOF, Autoencoder. Temporal features from historical buffer.

Configuration: contamination=0.02, min_votes=3 (majority).
Use use_ensemble=False to fall back to IF-only when not pretrained.

This detector runs conditionally (if enabled) as Layer 2.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = structlog.get_logger()

# NAB feature window (matches train_industry_anomaly.py add_features)
NAB_FEATURE_WINDOW = 10


class MLDetector:
    """
    4-Model weighted ensemble anomaly detector with temporal features.

    Uses Isolation Forest, OCSVM, LOF, and windowed Autoencoder with 3/4
    majority voting (weighted by per-model validation F1). Falls back to
    IF-only if use_ensemble=False.

    Attributes:
        use_ensemble: Use 4-model ensemble (default: True)
        contamination: Expected proportion of anomalies (0.02)
        min_votes: Minimum models that must agree for anomaly (default: 3)
        model: Primary model for compatibility (IF or ensemble)
        feature_names: List of metric names used as features
        domain: Domain scope for training
    """

    WINDOW_SIZE = 7  # Temporal window for autoencoder and rolling features

    def __init__(
        self,
        use_ensemble: bool = True,
        use_pretrained: bool = False,
        contamination: float = 0.02,
        n_estimators: int = 200,
        random_state: int = 42,
        min_votes: int = 3,
        pretrained_model_path: Optional[Path] = None,
    ):
        """
        Initialize the ML detector.

        Args:
            use_ensemble: Use 4-model ensemble when not pretrained (default: True)
            use_pretrained: Use pretrained NAB model instead of training (default: False)
            contamination: Expected proportion of anomalies (default: 0.02)
            n_estimators: Number of isolation trees for IF (default: 200)
            random_state: Random seed for reproducibility (default: 42)
            min_votes: Minimum models that must vote anomaly (default: 3)
            pretrained_model_path: Path to NAB joblib (default: models/anomaly_industry/isolation_forest_nab.joblib)
        """
        self.use_ensemble = use_ensemble
        self.use_pretrained = use_pretrained
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_votes = min_votes
        self.model: Optional[IsolationForest] = None  # Primary IF for compatibility
        self.feature_names: list[str] = []
        self._base_feature_names: list[str] = []  # Before temporal augmentation
        self.domain: Optional[str] = None
        self.logger = structlog.get_logger()

        # Pretrained NAB mode
        self._nab_model: Optional[IsolationForest] = None
        self._nab_scaler: Optional[StandardScaler] = None
        self._pretrained_model_path = pretrained_model_path

        # Ensemble components (used when use_ensemble=True and not pretrained)
        self._scaler: Optional[StandardScaler] = None
        self._models: dict[str, Any] = {}  # if_model, ocsvm, lof, autoencoder
        self._autoencoder_threshold: float = 0.0  # 99.5th pct of train recon error
        self._model_weights: dict[str, float] = {
            "if_model": 0.25, "ocsvm": 0.25, "lof": 0.25, "autoencoder": 0.25,
        }

        # Historical buffer for temporal features and windowed AE (or univariate for NAB)
        self._historical_buffer: list[np.ndarray] = []
        self._univariate_buffer: list[float] = []  # For NAB 5-feature computation
        self._buffer_maxlen: int = 30

    def train(
        self,
        historical_metrics: list[dict],
        domain: str,
        min_samples: int = 30,
    ) -> dict:
        """
        Train Isolation Forest model on historical metrics.

        Builds feature matrix from historical data and trains unsupervised
        anomaly detection model. Each sample represents one day's metrics.

        Args:
            historical_metrics: List of daily metric dictionaries
            domain: Domain to train on ("financial", "operational", "customer")
            min_samples: Minimum samples required for training (default: 30)

        Returns:
            Dictionary containing training metadata:
            {
                "success": bool,
                "samples_count": int,
                "features_count": int,
                "domain": str,
                "contamination": float,
                "message": str
            }

        Raises:
            ValueError: If insufficient training data

        Example:
            >>> historical = [
            ...     {"financial": {"refund_rate": 0.03, "margin_proxy": 0.25}},
            ...     {"financial": {"refund_rate": 0.02, "margin_proxy": 0.28}},
            ... ]
            >>> result = detector.train(historical, domain="financial")
        """
        self.logger.info(
            "training_ml_detector",
            domain=domain,
            historical_count=len(historical_metrics),
            min_samples=min_samples,
            use_pretrained=self.use_pretrained,
        )

        # Pretrained mode: load NAB model and populate univariate buffer (no training)
        if self.use_pretrained:
            return self._train_pretrained(historical_metrics, domain, min_samples)

        # Extract domain-specific metrics
        domain_metrics = []
        for metrics_dict in historical_metrics:
            if domain in metrics_dict:
                domain_metrics.append(metrics_dict[domain])
            else:
                # Assume flat structure if domain key not present
                domain_metrics.append(metrics_dict)

        if len(domain_metrics) < min_samples:
            message = f"Insufficient training data: {len(domain_metrics)} < {min_samples}"
            self.logger.warning(
                "insufficient_training_data",
                domain=domain,
                samples=len(domain_metrics),
                required=min_samples,
            )
            return {
                "success": False,
                "samples_count": len(domain_metrics),
                "features_count": 0,
                "domain": domain,
                "contamination": self.contamination,
                "message": message,
            }

        # Build feature matrix
        feature_matrix, feature_names = self._build_feature_matrix(domain_metrics)

        if feature_matrix.shape[0] == 0 or feature_matrix.shape[1] == 0:
            message = "No valid features extracted from metrics"
            self.logger.warning(
                "no_valid_features",
                domain=domain,
            )
            return {
                "success": False,
                "samples_count": 0,
                "features_count": 0,
                "domain": domain,
                "contamination": self.contamination,
                "message": message,
            }

        self._base_feature_names = feature_names
        self.domain = domain

        if self.use_ensemble:
            # Augment with temporal features (rolling mean/std/wow delta)
            augmented_matrix = self._augment_temporal(feature_matrix)
            temporal_names = []
            for fn in feature_names:
                temporal_names.extend([
                    f"{fn}_roll7_mean", f"{fn}_roll14_std", f"{fn}_wow_delta",
                ])
            self.feature_names = feature_names + temporal_names

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(augmented_matrix)

            # 1. Isolation Forest
            if_model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                max_features=0.8,
                random_state=self.random_state,
                n_jobs=-1,
            )
            if_model.fit(X_scaled)

            # 2. One-Class SVM (nu aligned with contamination)
            ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=self.contamination)
            ocsvm.fit(X_scaled)

            # 3. LOF
            lof = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1,
            )
            lof.fit(X_scaled)

            # 4. Windowed Autoencoder (MLPRegressor with 7-day flattened window)
            X_windowed = self._build_windows(X_scaled, self.WINDOW_SIZE)
            if X_windowed.shape[0] > 0:
                ae = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32, 64, 128),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    alpha=0.001,
                    random_state=self.random_state,
                )
                ae.fit(X_windowed, X_windowed)
                recon = ae.predict(X_windowed)
                recon_error = np.mean((X_windowed - recon) ** 2, axis=1)
                self._autoencoder_threshold = float(np.percentile(recon_error, 99.5))
            else:
                # Not enough data for windowed AE, fall back to flat
                ae = MLPRegressor(
                    hidden_layer_sizes=(64, 32, 16, 32, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    alpha=0.001,
                    random_state=self.random_state,
                )
                ae.fit(X_scaled, X_scaled)
                recon = ae.predict(X_scaled)
                recon_error = np.mean((X_scaled - recon) ** 2, axis=1)
                self._autoencoder_threshold = float(np.percentile(recon_error, 99.5))

            self._models = {
                "if_model": if_model,
                "ocsvm": ocsvm,
                "lof": lof,
                "autoencoder": ae,
            }
            self.model = if_model  # Primary for compatibility

            # Compute per-model pseudo-label weights on training data
            self._compute_model_weights(X_scaled)
        else:
            self.feature_names = feature_names
            # Single Isolation Forest (legacy)
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model.fit(feature_matrix)
            self._scaler = None
            self._models = {}

        self.logger.info(
            "ml_model_trained",
            domain=domain,
            samples=feature_matrix.shape[0],
            features=len(self.feature_names),
            contamination=self.contamination,
            use_ensemble=self.use_ensemble,
            min_votes=self.min_votes,
        )

        return {
            "success": True,
            "samples_count": feature_matrix.shape[0],
            "features_count": len(self.feature_names),
            "domain": domain,
            "contamination": self.contamination,
            "message": f"Model trained successfully on {feature_matrix.shape[0]} samples",
        }

    def detect(self, current_metrics: dict) -> dict:
        """
        Detect anomalies using trained model(s). With ensemble, uses weighted
        majority voting (min_votes threshold, default 3/4).
        With pretrained NAB, uses loaded model on 5-feature representation.

        Returns:
            Dictionary: is_anomaly, anomaly_score, prediction, confidence, features_used
        """
        if self._nab_model is not None:
            return self._detect_pretrained(current_metrics)
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.logger.debug(
            "running_ml_detection",
            domain=self.domain,
            metrics_count=len(current_metrics),
        )

        # Build base feature vector
        base_vector = self._build_feature_vector(current_metrics, self._base_feature_names or self.feature_names)

        if self.use_ensemble and self._models and self._scaler is not None:
            # Augment with temporal features from historical buffer
            temporal_vector = self._compute_temporal_vector(base_vector)
            full_vector = np.concatenate([base_vector, temporal_vector])
            feature_vector_2d = full_vector.reshape(1, -1)

            X = self._scaler.transform(feature_vector_2d)

            # Individual model predictions
            pred_if = int(self._models["if_model"].predict(X)[0])
            pred_ocsvm = int(self._models["ocsvm"].predict(X)[0])
            pred_lof = int(self._models["lof"].predict(X)[0])

            # Windowed autoencoder prediction
            pred_ae = self._predict_windowed_ae(X)

            # Weighted majority voting
            raw_votes = {
                "if_model": 1.0 if pred_if == -1 else 0.0,
                "ocsvm": 1.0 if pred_ocsvm == -1 else 0.0,
                "lof": 1.0 if pred_lof == -1 else 0.0,
                "autoencoder": 1.0 if pred_ae == -1 else 0.0,
            }
            weighted_score = sum(
                self._model_weights.get(k, 0.25) * v for k, v in raw_votes.items()
            )
            votes_anomaly = sum(1 for v in raw_votes.values() if v > 0)
            is_anomaly = votes_anomaly >= self.min_votes
            prediction = -1 if is_anomaly else 1
            anomaly_score = float(self._models["if_model"].score_samples(X)[0])

            # Update historical buffer
            self._historical_buffer.append(base_vector.copy())
            if len(self._historical_buffer) > self._buffer_maxlen:
                self._historical_buffer.pop(0)
        else:
            feature_vector_2d = base_vector.reshape(1, -1)
            prediction = int(self.model.predict(feature_vector_2d)[0])
            anomaly_score = float(self.model.score_samples(feature_vector_2d)[0])
            is_anomaly = prediction == -1

        confidence = abs(anomaly_score)
        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "features_used": self.feature_names,
        }

        if is_anomaly:
            self.logger.info(
                "ml_anomaly_detected",
                domain=self.domain,
                anomaly_score=result["anomaly_score"],
                confidence=result["confidence"],
            )

        return result

    def save_model(self, file_path: Path) -> bool:
        """Save trained model(s) to disk for MLflow registry."""
        if self.model is None:
            self.logger.warning("cannot_save_untrained_model")
            return False

        try:
            model_data: dict[str, Any] = {
                "model": self.model,
                "feature_names": self.feature_names,
                "base_feature_names": self._base_feature_names,
                "domain": self.domain,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "use_ensemble": self.use_ensemble,
                "min_votes": self.min_votes,
                "trained_at": datetime.utcnow().isoformat(),
            }
            if self.use_ensemble and self._models:
                model_data["scaler"] = self._scaler
                model_data["models"] = self._models
                model_data["autoencoder_threshold"] = self._autoencoder_threshold
                model_data["model_weights"] = self._model_weights

            with open(file_path, "wb") as f:
                pickle.dump(model_data, f)

            self.logger.info(
                "model_saved",
                file_path=str(file_path),
                domain=self.domain,
                use_ensemble=self.use_ensemble,
            )
            return True

        except Exception as e:
            self.logger.error(
                "model_save_failed",
                error=str(e),
                file_path=str(file_path),
            )
            return False

    def load_model(self, file_path: Path) -> bool:
        """Load trained model(s) from disk."""
        try:
            with open(file_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.feature_names = model_data["feature_names"]
            self._base_feature_names = model_data.get("base_feature_names", self.feature_names)
            self.domain = model_data["domain"]
            self.contamination = model_data.get("contamination", 0.02)
            self.n_estimators = model_data.get("n_estimators", 200)
            self.random_state = model_data.get("random_state", 42)
            self.min_votes = model_data.get("min_votes", 3)
            self.use_ensemble = "models" in model_data

            if self.use_ensemble:
                self._scaler = model_data.get("scaler")
                self._models = model_data["models"]
                self._autoencoder_threshold = model_data.get("autoencoder_threshold", 0.0)
                self._model_weights = model_data.get("model_weights", {
                    "if_model": 0.25, "ocsvm": 0.25, "lof": 0.25, "autoencoder": 0.25,
                })
            else:
                self._scaler = None
                self._models = {}

            self.logger.info(
                "model_loaded",
                file_path=str(file_path),
                domain=self.domain,
                use_ensemble=self.use_ensemble,
                trained_at=model_data.get("trained_at", "unknown"),
            )
            return True

        except Exception as e:
            self.logger.error(
                "model_load_failed",
                error=str(e),
                file_path=str(file_path),
            )
            return False

    # =========================================================================
    # Pretrained NAB Mode
    # =========================================================================

    def _train_pretrained(
        self,
        historical_metrics: list[dict],
        domain: str,
        min_samples: int,
    ) -> dict:
        """Load pretrained NAB model and populate univariate buffer. No training."""
        # Resolve model path
        path = self._pretrained_model_path
        if path is None:
            try:
                from api.config import get_settings
                models_dir = Path(get_settings().models_dir)
            except Exception:
                models_dir = Path("./models")
            path = models_dir / "anomaly_industry" / "isolation_forest_nab.joblib"

        if not path.exists():
            self.logger.warning(
                "pretrained_nab_not_found",
                path=str(path),
                message="Run: python scripts/train_industry_anomaly.py --model isolation_forest",
            )
            return {
                "success": False,
                "samples_count": 0,
                "features_count": 0,
                "domain": domain,
                "contamination": self.contamination,
                "message": f"Pretrained model not found at {path}",
            }

        try:
            artifact = joblib.load(path)
            self._nab_model = artifact["model"]
            self._nab_scaler = artifact["scaler"]
            self.model = self._nab_model  # For compatibility
            self.domain = domain
            self.feature_names = ["value", "rolling_mean", "rolling_std", "deviation", "rate_of_change"]

            # Populate univariate buffer from historical metrics
            univariate = self._metrics_to_univariate(historical_metrics)
            self._univariate_buffer = list(univariate[-self._buffer_maxlen:])

            self.logger.info(
                "pretrained_nab_loaded",
                path=str(path),
                buffer_samples=len(self._univariate_buffer),
            )
            return {
                "success": True,
                "samples_count": len(self._univariate_buffer),
                "features_count": 5,
                "domain": domain,
                "contamination": self.contamination,
                "message": f"Pretrained NAB model loaded from {path.name}",
            }
        except Exception as e:
            self.logger.error(
                "pretrained_nab_load_failed",
                path=str(path),
                error=str(e),
            )
            return {
                "success": False,
                "samples_count": 0,
                "features_count": 0,
                "domain": domain,
                "contamination": self.contamination,
                "message": str(e),
            }

    def _detect_pretrained(self, current_metrics: dict) -> dict:
        """Detect anomalies using pretrained NAB model."""
        univariate_val = self._metrics_to_univariate([current_metrics])[0]
        self._univariate_buffer.append(univariate_val)
        if len(self._univariate_buffer) > self._buffer_maxlen:
            self._univariate_buffer.pop(0)

        values = np.array(self._univariate_buffer)
        features = self._add_nab_features(values)
        X = features[-1:].reshape(1, -1)

        if self._nab_scaler is not None:
            X = self._nab_scaler.transform(X)

        prediction = int(self._nab_model.predict(X)[0])
        anomaly_score = float(self._nab_model.score_samples(X)[0])
        is_anomaly = prediction == -1
        confidence = abs(anomaly_score)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "features_used": self.feature_names,
        }

    @staticmethod
    def _metrics_to_univariate(metrics_list: list[dict]) -> np.ndarray:
        """
        Convert multivariate daily metrics to univariate proxy per day.
        Flattens nested dicts, takes mean of numeric values per day, normalizes 0-1.
        """
        if not metrics_list:
            return np.array([])

        def _extract_numerics(d: dict) -> list[float]:
            out = []
            for v in d.values():
                if isinstance(v, (int, float)) and v is not None:
                    out.append(float(v))
                elif isinstance(v, dict):
                    out.extend(_extract_numerics(v))
            return out

        all_values = []
        for m in metrics_list:
            if not isinstance(m, dict):
                all_values.append(0.0)
                continue
            vals = _extract_numerics(m)
            all_values.append(np.mean(vals) if vals else 0.0)

        arr = np.array(all_values)
        if len(arr) > 1:
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
        return arr

    @staticmethod
    def _add_nab_features(X: np.ndarray, window: int = NAB_FEATURE_WINDOW) -> np.ndarray:
        """
        Add rolling statistics as features (matches train_industry_anomaly.py add_features).
        Returns (n, 5): value, rolling_mean, rolling_std, deviation, rate_of_change.
        """
        values = X.flatten()
        n = len(values)

        features = np.zeros((n, 5))
        features[:, 0] = values

        for i in range(n):
            start = max(0, i - window)
            features[i, 1] = np.mean(values[start : i + 1])

        for i in range(n):
            start = max(0, i - window)
            segment = values[start : i + 1]
            features[i, 2] = np.std(segment) if len(segment) > 1 else 0

        features[:, 3] = values - features[:, 1]

        features[1:, 4] = np.diff(values)
        features[0, 4] = 0

        return features

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _build_feature_matrix(
        self, metrics_list: list[dict]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build feature matrix from list of metric dictionaries.

        Extracts all numeric metrics and builds a consistent feature matrix
        where each row is a day and each column is a metric.

        Args:
            metrics_list: List of daily metric dictionaries

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Collect all unique metric names
        all_metric_names = set()
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    all_metric_names.add(key)

        # Sort for consistency
        feature_names = sorted(all_metric_names)

        # Build matrix
        feature_matrix = []
        for metrics in metrics_list:
            row = self._build_feature_vector(metrics, feature_names)
            feature_matrix.append(row)

        return np.array(feature_matrix), feature_names

    def _build_feature_vector(
        self, metrics: dict, feature_names: list[str]
    ) -> np.ndarray:
        """
        Build feature vector from metrics dictionary.

        Extracts values in consistent order matching feature_names.
        Missing values are filled with 0.0.
        """
        vector = []
        for feature_name in feature_names:
            value = metrics.get(feature_name, 0.0)
            if value is None:
                value = 0.0
            vector.append(float(value))
        return np.array(vector)

    # =========================================================================
    # Temporal Feature Engineering
    # =========================================================================

    def _augment_temporal(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Augment feature matrix with temporal features: 7-day rolling mean,
        14-day rolling std, and week-over-week delta for each base feature.

        Args:
            feature_matrix: (N, F) array of base features

        Returns:
            (N, F*4) array with base + temporal features
        """
        n_samples, n_features = feature_matrix.shape
        augmented = []
        for i in range(n_samples):
            row = feature_matrix[i]
            window_7 = feature_matrix[max(0, i - 7):i] if i > 0 else row.reshape(1, -1)
            window_14 = feature_matrix[max(0, i - 14):i] if i > 0 else row.reshape(1, -1)
            row_7_ago = feature_matrix[i - 7] if i >= 7 else row

            mean_7 = window_7.mean(axis=0) if len(window_7) > 0 else np.zeros(n_features)
            std_14 = window_14.std(axis=0) if len(window_14) > 1 else np.zeros(n_features)
            wow_delta = row - row_7_ago

            augmented.append(np.concatenate([row, mean_7, std_14, wow_delta]))

        return np.array(augmented)

    def _compute_temporal_vector(self, base_vector: np.ndarray) -> np.ndarray:
        """
        Compute temporal features for a single observation using the historical buffer.

        Returns a vector with [roll7_mean, roll14_std, wow_delta] for each base feature.
        """
        n_features = len(base_vector)
        buf = self._historical_buffer

        if len(buf) >= 7:
            window_7 = np.array(buf[-7:])
            mean_7 = window_7.mean(axis=0)
        elif len(buf) > 0:
            mean_7 = np.array(buf).mean(axis=0)
        else:
            mean_7 = np.zeros(n_features)

        if len(buf) >= 14:
            window_14 = np.array(buf[-14:])
            std_14 = window_14.std(axis=0)
        elif len(buf) > 1:
            std_14 = np.array(buf).std(axis=0)
        else:
            std_14 = np.zeros(n_features)

        row_7_ago = buf[-7] if len(buf) >= 7 else base_vector
        wow_delta = base_vector - row_7_ago

        return np.concatenate([mean_7, std_14, wow_delta])

    # =========================================================================
    # Windowed Autoencoder
    # =========================================================================

    @staticmethod
    def _build_windows(X: np.ndarray, window_size: int) -> np.ndarray:
        """
        Build flattened sliding windows for the autoencoder.

        Args:
            X: (N, F) scaled feature matrix
            window_size: Number of consecutive rows per window

        Returns:
            (N - window_size, window_size * F) flattened windows
        """
        if len(X) <= window_size:
            return np.empty((0, window_size * X.shape[1]))
        rows = []
        for i in range(window_size, len(X)):
            window = X[i - window_size:i].flatten()
            rows.append(window)
        return np.array(rows)

    def _predict_windowed_ae(self, X_scaled: np.ndarray) -> int:
        """
        Predict anomaly using windowed autoencoder.

        Uses historical buffer to construct the window. Falls back to flat
        prediction if insufficient history.
        """
        ae = self._models.get("autoencoder")
        if ae is None:
            return 1  # No autoencoder, default to normal

        n_features = X_scaled.shape[1]
        expected_input = ae.n_features_in_

        # Check if AE was trained with windowed input
        if expected_input > n_features:
            # Windowed AE: build window from buffer + current
            window_rows = []
            for hist in self._historical_buffer[-(self.WINDOW_SIZE - 1):]:
                # Re-scale historical buffer entries
                hist_scaled = self._scaler.transform(
                    np.concatenate([hist, self._compute_temporal_vector(hist)]).reshape(1, -1)
                )
                window_rows.append(hist_scaled.flatten())

            window_rows.append(X_scaled.flatten())

            # Pad with zeros if not enough history
            while len(window_rows) < self.WINDOW_SIZE:
                window_rows.insert(0, np.zeros(n_features))

            # Take last WINDOW_SIZE rows and flatten
            window_rows = window_rows[-self.WINDOW_SIZE:]
            X_flat = np.concatenate(window_rows).reshape(1, -1)

            # Truncate or pad to match expected input size
            if X_flat.shape[1] > expected_input:
                X_flat = X_flat[:, :expected_input]
            elif X_flat.shape[1] < expected_input:
                pad = np.zeros((1, expected_input - X_flat.shape[1]))
                X_flat = np.concatenate([X_flat, pad], axis=1)
        else:
            # Flat AE (legacy or fallback)
            X_flat = X_scaled

        recon = ae.predict(X_flat)
        recon_err = float(np.mean((X_flat - recon) ** 2))
        return -1 if recon_err >= self._autoencoder_threshold else 1

    # =========================================================================
    # Weighted Voting
    # =========================================================================

    def _compute_model_weights(self, X_scaled: np.ndarray) -> None:
        """
        Compute per-model weights based on pseudo-label agreement F1.

        Uses top-contamination% of IF anomaly scores as pseudo ground truth,
        then evaluates each model's agreement. Models that better match the
        consensus get higher weights.
        """
        try:
            from sklearn.metrics import f1_score as sk_f1

            # Pseudo ground truth: top contamination% by IF anomaly score
            if_scores = self._models["if_model"].score_samples(X_scaled)
            threshold = np.percentile(if_scores, self.contamination * 100)
            y_pseudo = (if_scores <= threshold).astype(int)

            weights = {}
            for name, model in self._models.items():
                if name == "autoencoder":
                    recon = model.predict(X_scaled)
                    recon_err = np.mean((X_scaled - recon) ** 2, axis=1)
                    preds = (recon_err >= self._autoencoder_threshold).astype(int)
                else:
                    raw = model.predict(X_scaled)
                    preds = (raw == -1).astype(int)

                f1 = sk_f1(y_pseudo, preds, zero_division=0)
                weights[name] = max(f1, 0.1)  # Floor at 0.1 to avoid zeroing out

            # Normalize to sum to 1
            total = sum(weights.values())
            self._model_weights = {k: v / total for k, v in weights.items()}

            self.logger.info("model_weights_computed", weights=self._model_weights)
        except Exception as e:
            self.logger.warning("model_weight_computation_failed", error=str(e))
            self._model_weights = {
                "if_model": 0.25, "ocsvm": 0.25, "lof": 0.25, "autoencoder": 0.25,
            }
