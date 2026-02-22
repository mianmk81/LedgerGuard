"""
Support Ticket Sentiment Analysis (H6) - NLP Engineer Role.

Uses TF-IDF+LR model (FinancialPhraseBank) when available, else lexicon-based fallback.
Output: sentiment score, risk bucket, "Escalating frustration" flag.
"""

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import structlog

from api.config import get_settings
from api.models.events import CanonicalEvent
from api.storage.base import StorageBackend

logger = structlog.get_logger()

_SENTIMENT_MODEL_CACHE: Optional[object] = None


def _load_sentiment_model() -> Optional[object]:
    """Lazy-load sentiment model from models/sentiment_industry (train_industry_sentiment.py).

    Prefers LinearSVC (Macro F1 0.88, industry-level) over TF-IDF+LR (fallback).
    Both take raw text input; same predict() interface.
    """
    global _SENTIMENT_MODEL_CACHE
    if _SENTIMENT_MODEL_CACHE is not None:
        return _SENTIMENT_MODEL_CACHE
    try:
        import sys
        import types

        import joblib

        from api.engine.sentiment.text_features import text_length_features

        settings = get_settings()
        models_dir = Path(settings.models_dir) / "sentiment_industry"
        # Prefer LinearSVC (industry-level Macro F1 0.88); fallback to TF-IDF+LR
        for candidate in ["linear_svc_financial_sentiment.joblib", "tfidf_lr_financial_sentiment.joblib"]:
            path = models_dir / candidate
            if path.exists():
                # Older models were pickled with __main__.text_length_features; patch for compat
                orig_main = sys.modules.get("__main__")
                compat_main = types.ModuleType("__main__")
                compat_main.text_length_features = text_length_features
                sys.modules["__main__"] = compat_main
                try:
                    _SENTIMENT_MODEL_CACHE = joblib.load(path)
                    logger.info("sentiment_model_loaded", path=str(path))
                    return _SENTIMENT_MODEL_CACHE
                finally:
                    if orig_main is not None:
                        sys.modules["__main__"] = orig_main
    except Exception as e:
        logger.warning("sentiment_model_load_failed", error=str(e))
    return None


# Escalation/frustration lexicon (NLP: domain-specific keyword expansion)
FRUSTRATION_KEYWORDS = {
    "escalat", "frustrat", "angry", "angrier", "disappoint", "unacceptable",
    "refund", "cancel", "cancel my", "terrible", "worst", "horrible",
    "never again", "fed up", "sick of", "complaint", "complain",
    "unhappy", "furious", "outraged", "ridiculous", "absurd",
    "manager", "supervisor", "executive", "legal", "lawyer",
    "escalation", "escalate", "speak to", "speak with",
}
# Positive/neutral (dampen false positives)
NEUTRAL_POSITIVE = {
    "thank", "thanks", "great", "helpful", "resolved", "appreciate",
    "question", "inquiry", "info", "information",
}


class TicketSentimentAnalyzer:
    """
    Analyzes support ticket text for frustration/escalation risk.

    Lightweight lexicon-based approach; no transformer deps.
    Output: sentiment score, risk bucket, "Escalating frustration" flag.
    """

    def __init__(
        self,
        storage: StorageBackend,
        lookback_days: int = 30,
        top_n: int = 20,
    ):
        self.storage = storage
        self.lookback_days = lookback_days
        self.top_n = top_n

    def _get_text(self, event: CanonicalEvent) -> str:
        """Extract analyzable text from ticket event."""
        attrs = getattr(event, "attributes", {}) or {}
        parts = []
        if attrs.get("subject"):
            parts.append(str(attrs["subject"]))
        if attrs.get("description"):
            parts.append(str(attrs["description"]))
        return " ".join(parts).lower()

    def _score_sentiment(self, text: str) -> tuple[float, list[str]]:
        """
        Score sentiment 0-100 using ML model if available, else lexicon fallback.

        Returns:
            (score, matched_factors)
        """
        if not text or not text.strip():
            return 0.0, ["no_text"]

        model = _load_sentiment_model()
        if model is not None:
            try:
                import pandas as pd
                # Pipeline expects Series; label 0=negative, 1=neutral, 2=positive
                pred = model.predict(pd.Series([text]))[0]
                proba = model.predict_proba(pd.Series([text]))[0] if hasattr(model, "predict_proba") else None
                if proba is not None:
                    # Weighted score: negative=high risk, neutral=medium, positive=low
                    score = float(proba[0] * 85 + proba[1] * 40 + proba[2] * 10)
                else:
                    score = 85.0 if pred == 0 else (40.0 if pred == 1 else 10.0)
                label_str = ["negative", "neutral", "positive"][pred]
                return max(0, min(100, score)), [f"ml:{label_str}", "model:tfidf_lr_financial_sentiment"]
            except Exception as e:
                logger.warning("sentiment_ml_inference_failed", error=str(e))

        # Lexicon fallback
        text_lower = text.lower()
        score = 0.0
        matched = []
        for kw in FRUSTRATION_KEYWORDS:
            if kw in text_lower:
                score += 8.0
                matched.append(kw)
        for kw in NEUTRAL_POSITIVE:
            if kw in text_lower:
                score -= 3.0
                matched.append(f"neutral:{kw}")
        if len(text) < 20:
            score *= 0.7
        return max(0, min(100, score)), matched[:5]

    def analyze(
        self,
        target_date: Optional[date] = None,
    ) -> dict:
        """
        Analyze support ticket sentiment for escalation risk.

        Returns:
            {
                "tickets": [{
                    "entity_id": str,
                    "subject": str,
                    "sentiment_score": float,
                    "risk_bucket": "low"|"medium"|"high"|"escalating_frustration",
                    "factors": [str],
                    "created_at": str,
                }],
                "summary": {
                    "escalating_frustration_count": int,
                    "high_risk_count": int,
                    "avg_sentiment_score": float,
                }
            }
        """
        target = target_date or date.today()
        end_str = (target + timedelta(days=1)).isoformat()
        start_str = (target - timedelta(days=self.lookback_days)).isoformat()

        events = self.storage.read_canonical_events(
            event_type="support_ticket_opened",
            start_time=start_str,
            end_time=end_str,
            limit=500,
        )

        # Also check support_tickets source if event_type filter misses
        all_events = events
        if len(events) < 10:
            all_events = self.storage.read_canonical_events(
                start_time=start_str,
                end_time=end_str,
                limit=500,
            )
            all_events = [
                e for e in all_events
                if getattr(e, "event_type", None) and "support_ticket" in str(getattr(e.event_type, "value", ""))
            ]
            if not all_events:
                all_events = events

        results = []
        for e in all_events:
            if not isinstance(e, CanonicalEvent):
                continue
            text = self._get_text(e)
            score, factors = self._score_sentiment(text)
            attrs = getattr(e, "attributes", {}) or {}
            subject = attrs.get("subject", "(no subject)")[:80]
            evt_time = getattr(e, "event_time", None)
            created = evt_time.isoformat() if evt_time else ""

            if score < 15:
                bucket = "low"
            elif score < 35:
                bucket = "medium"
            elif score < 55:
                bucket = "high"
            else:
                bucket = "escalating_frustration"

            results.append({
                "entity_id": getattr(e, "entity_id", ""),
                "subject": subject,
                "sentiment_score": round(score, 1),
                "risk_bucket": bucket,
                "factors": factors,
                "created_at": created,
            })

        results.sort(key=lambda x: (-x["sentiment_score"], x["created_at"] or ""))
        top = results[: self.top_n]
        escalating = sum(1 for r in results if r["risk_bucket"] == "escalating_frustration")
        high = sum(1 for r in results if r["risk_bucket"] in ("high", "escalating_frustration"))
        avg_score = sum(r["sentiment_score"] for r in results) / len(results) if results else 0.0

        logger.info(
            "ticket_sentiment_computed",
            tickets_analyzed=len(results),
            escalating_frustration=escalating,
            high_risk=high,
        )

        return {
            "tickets": top,
            "summary": {
                "tickets_analyzed": len(results),
                "escalating_frustration_count": escalating,
                "high_risk_count": high,
                "avg_sentiment_score": round(avg_score, 2),
                "target_date": target.isoformat(),
            },
        }
