"""
Pipeline Configuration and Constants

Centralized settings for data paths, hyperparameters, and feature definitions.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# Define project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class PipelineConfig:
    """Configuration for the misinformation detection pipeline."""
    
    # Paths
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "fakenewsnet")
    processed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    
    # FakeNewsNet CSV files to load
    csv_files: Dict[str, tuple] = field(default_factory=lambda: {
        "politifact_real.csv": ("real", "politifact"),
        "politifact_fake.csv": ("fake", "politifact"),
        "gossipcop_real.csv": ("real", "gossipcop"),
        "gossipcop_fake.csv": ("fake", "gossipcop"),
    })
    
    # Data cleaning parameters
    min_title_length: int = 5  # Minimum characters in title
    remove_duplicates: bool = True
    dedup_column: str = "title"
    
    # Feature extraction parameters
    linguistic_features: List[str] = field(default_factory=lambda: [
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_negative",
        "subjectivity",
        "flesch_kincaid_grade",
        "ari",
        "lexical_diversity",
        "certainty_terms",
        "hedging_terms",
        "emotional_intensifiers",
    ])
    
    # Vectorization parameters
    tfidf_params: Dict = field(default_factory=lambda: {
        "stop_words": "english",
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_df": 0.8,
        "sublinear_tf": True,
    })
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Logging
    verbose: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str):
        """Log message if verbose is True."""
        if self.verbose:
            print(message)


# Linguistic markers for feature extraction
CERTAINTY_TERMS = [
    "always", "never", "definitely", "proven", "guaranteed",
    "certainly", "absolutely", "must", "will", "confirmed"
]

HEDGING_TERMS = [
    "might", "could", "allegedly", "reportedly", "suggests",
    "possibly", "may", "seems", "appears", "reported"
]

EMOTIONAL_INTENSIFIERS = [
    "shocking", "amazing", "incredible", "unbelievable", "devastating",
    "stunning", "horrifying", "heartbreaking", "outrageous"
]
