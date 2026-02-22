"""
Text feature extraction for sentiment models.

Shared between training (train_industry_sentiment.py) and inference (ticket_sentiment.py).
Must be in api/ so joblib can unpickle the pipeline that references this module.
"""

import numpy as np
import pandas as pd


def text_length_features(texts):
    """Extract text-length features from a Series of strings."""
    if isinstance(texts, pd.DataFrame):
        texts = texts.iloc[:, 0]
    result = pd.DataFrame({
        "char_count": texts.str.len().fillna(0),
        "word_count": texts.str.split().str.len().fillna(0),
        "avg_word_len": texts.apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) else 0
        ),
        "has_number": texts.str.contains(r"\d", regex=True, na=False).astype(int),
        "has_percent": texts.str.contains(r"%", na=False).astype(int),
        "uppercase_ratio": texts.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        ),
    })
    return result.values
