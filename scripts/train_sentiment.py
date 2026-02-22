"""
Sentiment Analysis Model Training for LedgerGuard.

Trains multiple sentiment analysis models on Olist e-commerce review data
(Portuguese reviews). Compares TF-IDF + Logistic Regression, Naive Bayes,
Random Forest, and Sentence Transformer + Logistic Regression classifiers.

Usage:
    python scripts/train_sentiment.py [--model tfidf_lr|naive_bayes|random_forest|transformer_lr|all]
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

# Check PyTorch version before importing sentence-transformers to avoid fatal crashes
# (sentence-transformers triggers deep import chains that can crash with incompatible versions)
SENTENCE_TRANSFORMERS_AVAILABLE = False
SentenceTransformer = None
try:
    import torch as _torch
    _torch_version = tuple(int(x) for x in _torch.__version__.split('.')[:2])
    if _torch_version < (2, 4):
        raise ImportError(f"PyTorch {_torch.__version__} < 2.4 required by sentence-transformers")
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except BaseException as e:
    print(f"\n[WARNING] sentence-transformers unavailable: {type(e).__name__}: {e}")
    print("  Transformer model will be skipped. Other models will train normally.")
    print("  To fix: pip install torch>=2.4 sentence-transformers\n")

warnings.filterwarnings('ignore')

logger = structlog.get_logger()

# Portuguese stopwords (common words to filter)
PORTUGUESE_STOPWORDS = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não",
    "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas",
    "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando",
    "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela",
    "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos",
    "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você",
    "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha",
    "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós",
    "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse",
    "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu",
    "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela",
    "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles",
    "aquelas", "isto", "aquilo", "estou", "está", "estamos", "estão",
    "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos",
    "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam",
    "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos",
    "estiverem", "hei", "há", "havemos", "hão", "houve", "houvemos",
    "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam",
    "houvesse", "houvéssemos", "houvessem", "houver", "houvermos",
    "houverem", "houverei", "houverá", "houveremos", "houverão", "houveria",
    "houveríamos", "houveriam", "sou", "somos", "são", "era", "éramos",
    "eram", "fui", "foi", "fomos", "foram", "fora", "fôramos", "seja",
    "sejamos", "sejam", "fosse", "fôssemos", "fossem", "for", "formos",
    "forem", "serei", "será", "seremos", "serão", "seria", "seríamos",
    "seriam", "tenho", "tem", "temos", "tém", "tinha", "tínhamos", "tinham",
    "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha",
    "tenhamos", "tenham", "tivesse", "tivéssemos", "tivessem", "tiver",
    "tivermos", "tiverem", "terei", "terá", "teremos", "terão", "teria",
    "teríamos", "teriam"
}


class SentimentTrainer:
    """
    Trains and evaluates sentiment analysis models on Olist review data.
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize sentiment trainer.

        Args:
            models_dir: Directory to save trained models
        """
        if models_dir is None:
            project_root = Path(__file__).parent.parent
            models_dir = project_root / "models" / "sentiment"

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger.bind(component="sentiment_trainer")
        self.results: Dict[str, Dict] = {}

    def _prepare_sentiment_data(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sentiment data from Olist reviews.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("preparing_sentiment_data")

        # Get project root and data directory
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data" / "olist"

        # Load reviews dataset
        reviews_path = data_dir / "olist_order_reviews_dataset.csv"
        if not reviews_path.exists():
            raise FileNotFoundError(
                f"Reviews dataset not found at {reviews_path}. "
                "Please download Olist dataset from Kaggle: "
                "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
            )

        reviews_df = pd.read_csv(reviews_path)

        # Extract review text and scores
        # Combine title and message for richer text features
        reviews_df['review_text'] = (
            reviews_df['review_comment_title'].fillna('') + ' ' +
            reviews_df['review_comment_message'].fillna('')
        ).str.strip()

        # Filter out reviews with no text
        reviews_df = reviews_df[reviews_df['review_text'].str.len() > 0]

        # Map scores to sentiment labels (1-5 star ratings)
        # 1-2 stars: negative (0)
        # 3 stars: neutral (1)
        # 4-5 stars: positive (2)
        def score_to_sentiment(score):
            if score <= 2:
                return 0  # negative
            elif score == 3:
                return 1  # neutral
            else:
                return 2  # positive

        reviews_df['sentiment'] = reviews_df['review_score'].apply(score_to_sentiment)

        # Shuffle the data
        reviews_df = reviews_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train (70%), val (15%), test (15%)
        n = len(reviews_df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_df = reviews_df.iloc[:train_end]
        val_df = reviews_df.iloc[train_end:val_end]
        test_df = reviews_df.iloc[val_end:]

        X_train = train_df['review_text'].values
        y_train = train_df['sentiment'].values

        X_val = val_df['review_text'].values
        y_val = val_df['sentiment'].values

        X_test = test_df['review_text'].values
        y_test = test_df['sentiment'].values

        self.logger.info(
            "sentiment_data_prepared",
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
            negative_count=int((reviews_df['sentiment'] == 0).sum()),
            neutral_count=int((reviews_df['sentiment'] == 1).sum()),
            positive_count=int((reviews_df['sentiment'] == 2).sum()),
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _extract_text_features(self, texts: np.ndarray) -> np.ndarray:
        """
        Extract text length features from raw text.

        Args:
            texts: Array of text strings

        Returns:
            Array with shape (n_samples, 4) containing:
            - text_length (character count)
            - word_count
            - exclamation_count
            - question_count
        """
        features = []
        for text in texts:
            text_str = str(text) if text is not None else ""
            features.append([
                len(text_str),  # text_length
                len(text_str.split()),  # word_count
                text_str.count('!'),  # exclamation_count
                text_str.count('?'),  # question_count
            ])
        return np.array(features)

    def train_tfidf_logistic_regression(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Train TF-IDF + text features + Logistic Regression model with balanced class weights.

        Args:
            X_train, X_val, X_test: Text features
            y_train, y_val, y_test: Sentiment labels

        Returns:
            Dictionary with model and metrics
        """
        model_name = "tfidf_lr"
        self.logger.info("training_model", model=model_name)

        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Create feature extraction functions
            text_length_extractor = FunctionTransformer(
                self._extract_text_features, validate=False
            )

            # Create feature union combining TF-IDF and text length features
            feature_union = FeatureUnion([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    stop_words=list(PORTUGUESE_STOPWORDS),
                )),
                ('text_features', text_length_extractor),
            ])

            # Create pipeline with balanced class weights
            pipeline = Pipeline([
                ('features', feature_union),
                ('classifier', LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    multi_class='multinomial',
                    class_weight='balanced',  # Handle class imbalance
                    random_state=42,
                    n_jobs=-1,
                ))
            ])

            # Log parameters
            mlflow.log_param("model_type", "tfidf_logistic_regression_enhanced")
            mlflow.log_param("max_features", 10000)
            mlflow.log_param("ngram_range", "(1, 2)")
            mlflow.log_param("C", 1.0)
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("extra_features", "text_length,word_count,exclamation_count,question_count")

            # Train
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training {model_name}...")
            pipeline.fit(X_train, y_train)

            # Evaluate
            metrics = self._evaluate_model(
                pipeline, X_train, X_val, X_test, y_train, y_val, y_test, model_name
            )

            # Save model
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(pipeline, model_path)
            self.logger.info("model_saved", path=str(model_path))

            # Log model to MLflow
            mlflow.sklearn.log_model(pipeline, "model")

            return {
                "model": pipeline,
                "metrics": metrics,
                "model_path": str(model_path),
            }

    def train_naive_bayes(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Train TF-IDF + Multinomial Naive Bayes model.

        Args:
            X_train, X_val, X_test: Text features
            y_train, y_val, y_test: Sentiment labels

        Returns:
            Dictionary with model and metrics
        """
        model_name = "naive_bayes"
        self.logger.info("training_model", model=model_name)

        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Create pipeline — use sample_weight for class balance (NB has no class_weight)
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    stop_words=list(PORTUGUESE_STOPWORDS),
                )),
                ('classifier', MultinomialNB(
                    alpha=0.1,
                    fit_prior=True,
                ))
            ])

            # Compute sample weights for class balance (NB has no class_weight param)
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

            # Log parameters
            mlflow.log_param("model_type", "tfidf_naive_bayes")
            mlflow.log_param("max_features", 10000)
            mlflow.log_param("ngram_range", "(1, 2)")
            mlflow.log_param("alpha", 0.1)

            # Train with sample weights
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training {model_name}...")
            pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

            # Evaluate
            metrics = self._evaluate_model(
                pipeline, X_train, X_val, X_test, y_train, y_val, y_test, model_name
            )

            # Save model
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(pipeline, model_path)
            self.logger.info("model_saved", path=str(model_path))

            # Log model to MLflow
            mlflow.sklearn.log_model(pipeline, "model")

            return {
                "model": pipeline,
                "metrics": metrics,
                "model_path": str(model_path),
            }

    def train_random_forest(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Train TF-IDF + text features + Random Forest model.

        Args:
            X_train, X_val, X_test: Text features
            y_train, y_val, y_test: Sentiment labels

        Returns:
            Dictionary with model and metrics
        """
        model_name = "random_forest"
        self.logger.info("training_model", model=model_name)

        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Create feature extraction functions
            text_length_extractor = FunctionTransformer(
                self._extract_text_features, validate=False
            )

            # Create feature union combining TF-IDF and text length features
            feature_union = FeatureUnion([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,  # Reduced for random forest
                    ngram_range=(1, 2),
                    min_df=3,
                    stop_words=list(PORTUGUESE_STOPWORDS),
                )),
                ('text_features', text_length_extractor),
            ])

            # Create pipeline — class_weight for 3-class (Neutral is hardest)
            pipeline = Pipeline([
                ('features', feature_union),
                ('classifier', RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ))
            ])

            # Log parameters
            mlflow.log_param("model_type", "tfidf_random_forest_enhanced")
            mlflow.log_param("max_features", 5000)
            mlflow.log_param("ngram_range", "(1, 2)")
            mlflow.log_param("n_estimators", 300)
            mlflow.log_param("max_depth", 20)
            mlflow.log_param("min_samples_leaf", 2)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("extra_features", "text_length,word_count,exclamation_count,question_count")

            # Train
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training {model_name}...")
            pipeline.fit(X_train, y_train)

            # Evaluate
            metrics = self._evaluate_model(
                pipeline, X_train, X_val, X_test, y_train, y_val, y_test, model_name
            )

            # Save model
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(pipeline, model_path)
            self.logger.info("model_saved", path=str(model_path))

            # Log model to MLflow
            mlflow.sklearn.log_model(pipeline, "model")

            return {
                "model": pipeline,
                "metrics": metrics,
                "model_path": str(model_path),
            }

    def train_transformer_logistic_regression(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Train Sentence Transformer embeddings + Logistic Regression model.

        Uses paraphrase-multilingual-MiniLM-L12-v2 for 384-dim embeddings.
        This model supports Portuguese and should significantly outperform bag-of-words.

        Args:
            X_train, X_val, X_test: Text features
            y_train, y_val, y_test: Sentiment labels

        Returns:
            Dictionary with model and metrics
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for this model. "
                "Install with: pip install sentence-transformers"
            )

        model_name = "transformer_lr"
        self.logger.info("training_model", model=model_name)

        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training {model_name}...")
            print("  Loading sentence transformer model (this may take a moment)...")

            # Load sentence transformer model
            transformer_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            # Log parameters
            mlflow.log_param("model_type", "sentence_transformer_logistic_regression")
            mlflow.log_param("transformer_model", "paraphrase-multilingual-MiniLM-L12-v2")
            mlflow.log_param("embedding_dim", 384)
            mlflow.log_param("C", 1.0)
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("class_weight", "balanced")

            # Encode texts into embeddings
            print(f"  Encoding training texts ({len(X_train)} samples)...")
            X_train_embeddings = transformer_model.encode(
                X_train.tolist(), show_progress_bar=True, batch_size=32
            )

            print(f"  Encoding validation texts ({len(X_val)} samples)...")
            X_val_embeddings = transformer_model.encode(
                X_val.tolist(), show_progress_bar=True, batch_size=32
            )

            print(f"  Encoding test texts ({len(X_test)} samples)...")
            X_test_embeddings = transformer_model.encode(
                X_test.tolist(), show_progress_bar=True, batch_size=32
            )

            # Train logistic regression on embeddings
            print("  Training logistic regression classifier...")
            classifier = LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class='multinomial',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )
            classifier.fit(X_train_embeddings, y_train)

            # Create a simple wrapper for prediction
            class TransformerPipeline:
                """Wrapper to encode text and predict in one step."""
                def __init__(self, transformer, classifier):
                    self.transformer = transformer
                    self.classifier = classifier

                def predict(self, X):
                    embeddings = self.transformer.encode(
                        X.tolist() if isinstance(X, np.ndarray) else X,
                        show_progress_bar=False,
                        batch_size=32
                    )
                    return self.classifier.predict(embeddings)

                def predict_proba(self, X):
                    embeddings = self.transformer.encode(
                        X.tolist() if isinstance(X, np.ndarray) else X,
                        show_progress_bar=False,
                        batch_size=32
                    )
                    return self.classifier.predict_proba(embeddings)

            pipeline = TransformerPipeline(transformer_model, classifier)

            # Evaluate using pre-computed embeddings for efficiency
            print("  Evaluating model...")
            metrics = self._evaluate_transformer_model(
                transformer_model,
                classifier,
                X_train_embeddings,
                X_val_embeddings,
                X_test_embeddings,
                y_train,
                y_val,
                y_test,
                model_name,
            )

            # Save model
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump({
                'transformer': transformer_model,
                'classifier': classifier,
            }, model_path)
            self.logger.info("model_saved", path=str(model_path))

            # Log classifier to MLflow (transformer is too large)
            mlflow.sklearn.log_model(classifier, "classifier")

            return {
                "model": pipeline,
                "metrics": metrics,
                "model_path": str(model_path),
            }

    def _evaluate_transformer_model(
        self,
        transformer_model,
        classifier,
        X_train_embeddings: np.ndarray,
        X_val_embeddings: np.ndarray,
        X_test_embeddings: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> Dict:
        """
        Evaluate transformer model using pre-computed embeddings.

        Args:
            transformer_model: Sentence transformer model
            classifier: Trained classifier
            X_train_embeddings, X_val_embeddings, X_test_embeddings: Pre-computed embeddings
            y_train, y_val, y_test: Label sets
            model_name: Name of the model

        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions on embeddings
        y_train_pred = classifier.predict(X_train_embeddings)
        y_val_pred = classifier.predict(X_val_embeddings)
        y_test_pred = classifier.predict(X_test_embeddings)

        # Calculate metrics
        metrics = {
            "train": {
                "accuracy": accuracy_score(y_train, y_train_pred),
                "macro_f1": f1_score(y_train, y_train_pred, average='macro'),
                "weighted_f1": f1_score(y_train, y_train_pred, average='weighted'),
            },
            "val": {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "macro_f1": f1_score(y_val, y_val_pred, average='macro'),
                "weighted_f1": f1_score(y_val, y_val_pred, average='weighted'),
            },
            "test": {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "macro_f1": f1_score(y_test, y_test_pred, average='macro'),
                "weighted_f1": f1_score(y_test, y_test_pred, average='weighted'),
            },
            "confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "classification_report": classification_report(
                y_test, y_test_pred, target_names=['Negative', 'Neutral', 'Positive']
            ),
        }

        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", metrics["train"]["accuracy"])
        mlflow.log_metric("val_accuracy", metrics["val"]["accuracy"])
        mlflow.log_metric("test_accuracy", metrics["test"]["accuracy"])
        mlflow.log_metric("test_macro_f1", metrics["test"]["macro_f1"])
        mlflow.log_metric("test_weighted_f1", metrics["test"]["weighted_f1"])

        # Print results
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model_name.upper()} Results:")
        print(f"  Train Accuracy: {metrics['train']['accuracy']:.4f}")
        print(f"  Val Accuracy:   {metrics['val']['accuracy']:.4f}")
        print(f"  Test Accuracy:  {metrics['test']['accuracy']:.4f}")
        print(f"  Test Macro F1:  {metrics['test']['macro_f1']:.4f}")
        print(f"  Test Weighted F1: {metrics['test']['weighted_f1']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        return metrics

    def _evaluate_model(
        self,
        model: Pipeline,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> Dict:
        """
        Evaluate model on train, val, and test sets.

        Args:
            model: Trained model pipeline
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Label sets
            model_name: Name of the model

        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "train": {
                "accuracy": accuracy_score(y_train, y_train_pred),
                "macro_f1": f1_score(y_train, y_train_pred, average='macro'),
                "weighted_f1": f1_score(y_train, y_train_pred, average='weighted'),
            },
            "val": {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "macro_f1": f1_score(y_val, y_val_pred, average='macro'),
                "weighted_f1": f1_score(y_val, y_val_pred, average='weighted'),
            },
            "test": {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "macro_f1": f1_score(y_test, y_test_pred, average='macro'),
                "weighted_f1": f1_score(y_test, y_test_pred, average='weighted'),
            },
            "confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "classification_report": classification_report(
                y_test, y_test_pred, target_names=['Negative', 'Neutral', 'Positive']
            ),
        }

        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", metrics["train"]["accuracy"])
        mlflow.log_metric("val_accuracy", metrics["val"]["accuracy"])
        mlflow.log_metric("test_accuracy", metrics["test"]["accuracy"])
        mlflow.log_metric("test_macro_f1", metrics["test"]["macro_f1"])
        mlflow.log_metric("test_weighted_f1", metrics["test"]["weighted_f1"])

        # Print results
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model_name.upper()} Results:")
        print(f"  Train Accuracy: {metrics['train']['accuracy']:.4f}")
        print(f"  Val Accuracy:   {metrics['val']['accuracy']:.4f}")
        print(f"  Test Accuracy:  {metrics['test']['accuracy']:.4f}")
        print(f"  Test Macro F1:  {metrics['test']['macro_f1']:.4f}")
        print(f"  Test Weighted F1: {metrics['test']['weighted_f1']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        return metrics

    def generate_comparison_report(self) -> None:
        """
        Generate comparison table and confusion matrix plots for all models.
        """
        if not self.results:
            self.logger.warning("no_results_to_compare")
            return

        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)

        # Create comparison table with better display names
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']['test']

            # Create more readable model names
            display_name = model_name.replace('_', ' ').title()
            if model_name == "transformer_lr":
                display_name = "Sentence Transformer + LR"
            elif model_name == "tfidf_lr":
                display_name = "TF-IDF + LR (Enhanced)"
            elif model_name == "naive_bayes":
                display_name = "TF-IDF + Naive Bayes"
            elif model_name == "random_forest":
                display_name = "TF-IDF + Random Forest"

            comparison_data.append({
                'Model': display_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Macro F1': f"{metrics['macro_f1']:.4f}",
                'Weighted F1': f"{metrics['weighted_f1']:.4f}",
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by Macro F1 (descending) to show best model first
        comparison_df['_sort_key'] = comparison_df['Macro F1'].astype(float)
        comparison_df = comparison_df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)

        print("\n" + comparison_df.to_string(index=False))

        # Save comparison table
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = reports_dir / "sentiment_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison table saved to: {comparison_path}")

        # Generate confusion matrix plots
        self._plot_confusion_matrices(reports_dir)

    def _plot_confusion_matrices(self, reports_dir: Path) -> None:
        """
        Plot confusion matrices for all models in a single figure.

        Args:
            reports_dir: Directory to save the plot
        """
        n_models = len(self.results)
        if n_models == 0:
            return

        # Calculate grid dimensions for better layout with 4 models
        if n_models <= 2:
            nrows, ncols = 1, n_models
            figsize = (6 * n_models, 5)
        elif n_models <= 4:
            nrows, ncols = 1, n_models
            figsize = (5 * n_models, 5)
        else:
            nrows = (n_models + 2) // 3  # Ceiling division
            ncols = min(3, n_models)
            figsize = (5 * ncols, 5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['metrics']['confusion_matrix']

            # Create more readable model names
            display_name = model_name.replace("_", " ").title()
            if model_name == "transformer_lr":
                display_name = "Sentence Transformer + LR"
            elif model_name == "tfidf_lr":
                display_name = "TF-IDF + LR (Enhanced)"

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[idx],
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
            )
            axes[idx].set_title(f'{display_name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        # Hide extra subplots if any
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        plot_path = reports_dir / "sentiment_confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {plot_path}")

    def train_all_models(self) -> None:
        """
        Train all sentiment analysis models.
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting sentiment analysis training")
        print("=" * 80)

        # Set MLflow experiment
        mlflow.set_experiment("ledgerguard-sentiment-analysis")

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_sentiment_data()

        # Train models
        models_to_train = {
            'tfidf_lr': self.train_tfidf_logistic_regression,
            'naive_bayes': self.train_naive_bayes,
            'random_forest': self.train_random_forest,
        }

        # Add transformer model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            models_to_train['transformer_lr'] = self.train_transformer_logistic_regression
        else:
            print("\n[WARNING] Skipping transformer_lr model (sentence-transformers not installed)")

        for model_name, train_func in models_to_train.items():
            try:
                result = train_func(X_train, X_val, X_test, y_train, y_val, y_test)
                self.results[model_name] = result
            except Exception as e:
                self.logger.error("model_training_failed", model=model_name, error=str(e))
                print(f"\n[ERROR] Failed to train {model_name}: {e}")

        # Generate comparison report
        self.generate_comparison_report()

        print("\n" + "=" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed!")
        print(f"Models saved to: {self.models_dir}")
        print("=" * 80)


def main():
    """
    Main entry point for sentiment training script.
    """
    parser = argparse.ArgumentParser(
        description="Train sentiment analysis models for LedgerGuard"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['tfidf_lr', 'naive_bayes', 'random_forest', 'transformer_lr', 'all'],
        help='Model to train (default: all)',
    )

    args = parser.parse_args()

    # Check if transformer model requested but not available
    if args.model == 'transformer_lr' and not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n[ERROR] transformer_lr model requires sentence-transformers")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    # Initialize trainer
    trainer = SentimentTrainer()

    # Set MLflow experiment
    mlflow.set_experiment("ledgerguard-sentiment-analysis")

    try:
        if args.model == 'all':
            trainer.train_all_models()
        else:
            # Train single model
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training {args.model}")
            print("=" * 80)

            X_train, X_val, X_test, y_train, y_val, y_test = trainer._prepare_sentiment_data()

            if args.model == 'tfidf_lr':
                result = trainer.train_tfidf_logistic_regression(
                    X_train, X_val, X_test, y_train, y_val, y_test
                )
            elif args.model == 'naive_bayes':
                result = trainer.train_naive_bayes(
                    X_train, X_val, X_test, y_train, y_val, y_test
                )
            elif args.model == 'random_forest':
                result = trainer.train_random_forest(
                    X_train, X_val, X_test, y_train, y_val, y_test
                )
            elif args.model == 'transformer_lr':
                result = trainer.train_transformer_logistic_regression(
                    X_train, X_val, X_test, y_train, y_val, y_test
                )

            trainer.results[args.model] = result
            trainer.generate_comparison_report()

            print("\n" + "=" * 80)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed!")
            print(f"Model saved to: {result['model_path']}")
            print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease download the Olist dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
        print(f"\nPlace the CSV files in: {Path(__file__).parent.parent / 'data' / 'olist'}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
