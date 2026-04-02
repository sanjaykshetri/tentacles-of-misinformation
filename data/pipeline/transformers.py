"""
Feature Transformation Module

Extracts linguistic and behavioral features from text.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

from .config import PipelineConfig, CERTAINTY_TERMS, HEDGING_TERMS, EMOTIONAL_INTENSIFIERS


class FeatureTransformer:
    """Extract linguistic features from article titles."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the feature transformer.
        
        Parameters
        ----------
        config : PipelineConfig, optional
            Pipeline configuration. If None, uses defaults.
        """
        self.config = config or PipelineConfig()
        
        # Initialize analyzers
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check and warn about missing dependencies."""
        if not VADER_AVAILABLE:
            self.config.log("Warning: vaderSentiment not installed. Sentiment analysis will be skipped.")
            self.config.log("  → pip install vaderSentiment")
        if not TEXTBLOB_AVAILABLE:
            self.config.log("Warning: textblob not installed. Subjectivity analysis will be skipped.")
            self.config.log("  → pip install textblob")
        if not TEXTSTAT_AVAILABLE:
            self.config.log("Warning: textstat not installed. Readability analysis will be skipped.")
            self.config.log("  → pip install textstat")
    
    def transform(self, df: pd.DataFrame, text_column: str = "title") -> pd.DataFrame:
        """
        Extract linguistic features from text.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with text column
        text_column : str
            Name of column containing text
        
        Returns
        -------
        pd.DataFrame
            Dataframe with extracted features
        """
        self.config.log("\n" + "=" * 70)
        self.config.log("FEATURE EXTRACTION")
        self.config.log("=" * 70)
        self.config.log(f"\nExtracting features from {len(df)} texts...")
        
        feats = pd.DataFrame(index=df.index)
        
        # Basic text features
        self.config.log("  • Basic text features...")
        feats["title_length_words"] = df[text_column].str.split().str.len()
        feats["title_length_chars"] = df[text_column].str.len()
        
        # Sentiment features
        if VADER_AVAILABLE:
            self.config.log("  • VADER sentiment analysis...")
            sentiment_scores = df[text_column].apply(self._get_sentiment_scores)
            feats["sentiment_compound"] = sentiment_scores.apply(lambda x: x["compound"])
            feats["sentiment_positive"] = sentiment_scores.apply(lambda x: x["pos"])
            feats["sentiment_negative"] = sentiment_scores.apply(lambda x: x["neg"])
        
        # Subjectivity
        if TEXTBLOB_AVAILABLE:
            self.config.log("  • TextBlob subjectivity...")
            feats["subjectivity"] = df[text_column].apply(self._get_subjectivity)
        
        # Readability
        if TEXTSTAT_AVAILABLE:
            self.config.log("  • Readability metrics...")
            readability = df[text_column].apply(self._get_readability)
            feats["flesch_kincaid_grade"] = readability.apply(lambda x: x["flesch_kincaid_grade"])
            feats["ari"] = readability.apply(lambda x: x["ari"])
        
        # Lexical diversity
        self.config.log("  • Lexical diversity...")
        feats["lexical_diversity"] = df[text_column].apply(self._lexical_diversity)
        
        # Language markers
        self.config.log("  • Certainty & hedging language...")
        feats["certainty_terms"] = df[text_column].apply(lambda x: self._count_terms(x, CERTAINTY_TERMS))
        feats["hedging_terms"] = df[text_column].apply(lambda x: self._count_terms(x, HEDGING_TERMS))
        feats["emotional_intensifiers"] = df[text_column].apply(lambda x: self._count_terms(x, EMOTIONAL_INTENSIFIERS))
        feats["certainty_hedging_ratio"] = feats["certainty_terms"] / (feats["hedging_terms"] + 1)
        
        self.config.log(f"✅ Extracted {feats.shape[1]} features\n")
        
        return feats
    
    @staticmethod
    def _lexical_diversity(text: str) -> float:
        """Type-Token Ratio: unique words / total words."""
        if not text or len(text) == 0:
            return 0.0
        words = text.lower().split()
        return len(set(words)) / max(len(words), 1)
    
    @staticmethod
    def _count_terms(text: str, term_list: list) -> int:
        """Count occurrences of terms in text."""
        if not text or len(text) == 0:
            return 0
        text_lower = text.lower()
        return sum(text_lower.count(term) for term in term_list)
    
    def _get_sentiment_scores(self, text: str) -> dict:
        """VADER sentiment analysis."""
        if not text or len(text) == 0 or not self.analyzer:
            return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
        try:
            return self.analyzer.polarity_scores(text)
        except Exception:
            return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
    
    @staticmethod
    def _get_subjectivity(text: str) -> float:
        """TextBlob subjectivity (0=objective, 1=subjective)."""
        if not text or len(text) == 0:
            return 0.0
        try:
            blob = TextBlob(text)
            return blob.sentiment.subjectivity
        except Exception:
            return 0.0
    
    @staticmethod
    def _get_readability(text: str) -> dict:
        """Flesch-Kincaid grade level and ARI."""
        if not text or len(text) == 0:
            return {"flesch_kincaid_grade": 0.0, "ari": 0.0}
        try:
            fk = textstat.flesch_kincaid_grade(text)
            ari = textstat.automated_readability_index(text)
            return {"flesch_kincaid_grade": fk, "ari": ari}
        except Exception:
            return {"flesch_kincaid_grade": 0.0, "ari": 0.0}
