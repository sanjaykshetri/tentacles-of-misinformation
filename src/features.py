"""
Sprint 3: Linguistic & Behavioral Feature Extraction

Extracts psychologically-grounded language features from article titles:
- Sentiment (VADER compound)
- Subjectivity (TextBlob)
- Readability (Flesch-Kincaid, ARI)
- Lexical diversity (Type-Token Ratio)
- Certainty vs Hedging language
"""

import pandas as pd
import numpy as np
import textstat
from pathlib import Path

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Install: pip install vaderSentiment")
    
try:
    from textblob import TextBlob
except ImportError:
    print("Install: pip install textblob")

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Linguistic markers
CERTAINTY_TERMS = [
    "always", "never", "definitely", "proven", "guaranteed",
    "certainly", "absolutely", "must", "will", "confirmed"
]

HEDGING_TERMS = [
    "might", "could", "allegedly", "reportedly", "suggests",
    "possibly", "may", "seems", "appears", "reportedly"
]

EMOTIONAL_INTENSIFIERS = [
    "shocking", "amazing", "incredible", "unbelievable", "devastating",
    "stunning", "horrifying", "heartbreaking", "outrageous", "stunning"
]

def lexical_diversity(text):
    """Type-Token Ratio: unique words / total words."""
    if not text or len(text) == 0:
        return 0.0
    words = text.lower().split()
    return len(set(words)) / max(len(words), 1)

def count_terms(text, term_list):
    """Count occurrences of terms in text (case-insensitive)."""
    if not text or len(text) == 0:
        return 0
    text_lower = text.lower()
    return sum(text_lower.count(term) for term in term_list)

def get_sentiment_scores(text):
    """VADER sentiment analysis."""
    if not text or len(text) == 0:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
    try:
        scores = analyzer.polarity_scores(text)
        return scores
    except:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}

def get_subjectivity(text):
    """TextBlob subjectivity (0=objective, 1=subjective)."""
    if not text or len(text) == 0:
        return 0.0
    try:
        blob = TextBlob(text)
        return blob.sentiment.subjectivity
    except:
        return 0.0

def get_readability(text):
    """Flesch-Kincaid grade level and ARI."""
    if not text or len(text) == 0:
        return {"flesch_kincaid_grade": 0.0, "ari": 0.0}
    try:
        fk = textstat.flesch_kincaid_grade(text)
        ari = textstat.automated_readability_index(text)
        return {"flesch_kincaid_grade": fk, "ari": ari}
    except:
        return {"flesch_kincaid_grade": 0.0, "ari": 0.0}

def extract_features(df, text_column="title"):
    """
    Extract linguistic features from text.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text column
    text_column : str
        Name of column containing text (default: "title")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted features
    """
    print(f"\nExtracting features from {len(df)} texts...")
    
    feats = pd.DataFrame(index=df.index)
    
    # Sentiment
    print("  • VADER sentiment...")
    sentiment_scores = df[text_column].apply(get_sentiment_scores)
    feats["sentiment_compound"] = sentiment_scores.apply(lambda x: x["compound"])
    feats["sentiment_positive"] = sentiment_scores.apply(lambda x: x["pos"])
    feats["sentiment_negative"] = sentiment_scores.apply(lambda x: x["neg"])
    
    # Subjectivity
    print("  • TextBlob subjectivity...")
    feats["subjectivity"] = df[text_column].apply(get_subjectivity)
    
    # Readability
    print("  • Readability metrics...")
    readability = df[text_column].apply(get_readability)
    feats["flesch_kincaid_grade"] = readability.apply(lambda x: x["flesch_kincaid_grade"])
    feats["ari"] = readability.apply(lambda x: x["ari"])
    
    # Lexical diversity
    print("  • Lexical diversity...")
    feats["lexical_diversity"] = df[text_column].apply(lexical_diversity)
    
    # Certainty vs Hedging
    print("  • Certainty & hedging language...")
    feats["certainty_terms"] = df[text_column].apply(lambda x: count_terms(x, CERTAINTY_TERMS))
    feats["hedging_terms"] = df[text_column].apply(lambda x: count_terms(x, HEDGING_TERMS))
    feats["emotional_intensifiers"] = df[text_column].apply(lambda x: count_terms(x, EMOTIONAL_INTENSIFIERS))
    
    # Ratios
    feats["certainty_hedging_ratio"] = feats["certainty_terms"] / (feats["hedging_terms"] + 1)
    
    print(f"✅ Extracted {feats.shape[1]} features\n")
    
    return feats

def generate_feature_report(feats):
    """Print summary statistics for features."""
    print("\n" + "="*70)
    print("FEATURE SUMMARY STATISTICS")
    print("="*70)
    
    print("\nSentiment Features:")
    print(feats[["sentiment_compound", "sentiment_positive", "sentiment_negative"]].describe().round(3))
    
    print("\n\nSubjectivity & Readability:")
    print(feats[["subjectivity", "flesch_kincaid_grade", "ari"]].describe().round(3))
    
    print("\n\nLexical & Language Features:")
    print(feats[["lexical_diversity", "certainty_terms", "hedging_terms", "emotional_intensifiers"]].describe().round(3))
    
    print("\n\nCertainty-Hedging Ratio:")
    print(feats["certainty_hedging_ratio"].describe().round(3))

if __name__ == "__main__":
    print("\n" + "🧠"*10)
    print("FEATURE EXTRACTION PIPELINE")
    print("🧠"*10)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "processed" / "articles.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Loaded {len(df)} articles")
    
    # Extract features
    feats = extract_features(df, text_column="title")
    
    # Report
    generate_feature_report(feats)
    
    # Save
    output_path = Path(__file__).parent.parent / "data" / "processed" / "features.parquet"
    feats.to_parquet(output_path)
    print(f"\n✅ Features saved to {output_path}")
    print(f"Shape: {feats.shape}")
    
    print("\n" + "✨"*10)
