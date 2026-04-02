"""
Data Cleaning Module

Handles data validation, deduplication, and quality checks.
"""

import pandas as pd
from typing import Optional
from .config import PipelineConfig


class DataCleaner:
    """Clean and validate misinformation dataset."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the data cleaner.
        
        Parameters
        ----------
        config : PipelineConfig, optional
            Pipeline configuration. If None, uses defaults.
        """
        self.config = config or PipelineConfig()
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset: remove nulls, duplicates, and short titles.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe to clean
        
        Returns
        -------
        pd.DataFrame
            Cleaned dataframe
        """
        self.config.log("\n" + "=" * 70)
        self.config.log("DATA CLEANING")
        self.config.log("=" * 70)
        
        df = df.copy()
        initial_size = len(df)
        
        # Step 1: Remove missing titles
        self.config.log(f"\nInitial dataset size: {len(df)} articles")
        df = df.dropna(subset=["title"])
        self.config.log(f"After removing NaN titles: {len(df)} articles")
        
        # Step 2: Strip whitespace
        df["title"] = df["title"].str.strip()
        
        # Step 3: Remove empty strings
        df = df[df["title"].str.len() > 0]
        self.config.log(f"After removing empty titles: {len(df)} articles")
        
        # Step 4: Remove very short titles
        df = df[df["title"].str.len() >= self.config.min_title_length]
        self.config.log(f"After removing short titles (< {self.config.min_title_length} chars): {len(df)} articles")
        
        # Step 5: Remove duplicates
        if self.config.remove_duplicates:
            df = df.drop_duplicates(subset=[self.config.dedup_column], keep="first")
            self.config.log(f"After deduplication: {len(df)} articles")
        
        # Calculate statistics
        removed = initial_size - len(df)
        pct_removed = (removed / initial_size * 100) if initial_size > 0 else 0
        
        self.config.log(f"\n✅ Cleaning complete")
        self.config.log(f"   Removed: {removed} articles ({pct_removed:.1f}%)")
        self.config.log(f"   Retained: {len(df)} articles ({100-pct_removed:.1f}%)")
        self.config.log(f"   Label distribution:\n{df['label'].value_counts()}\n")
        
        return df
    
    def get_quality_report(self, df: pd.DataFrame) -> dict:
        """
        Generate quality metrics for the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to analyze
        
        Returns
        -------
        dict
            Quality metrics
        """
        report = {
            "total_articles": len(df),
            "missing_titles": df["title"].isna().sum(),
            "duplicate_titles": df["title"].duplicated().sum(),
            "avg_title_length": df["title"].str.len().mean(),
            "min_title_length": df["title"].str.len().min(),
            "max_title_length": df["title"].str.len().max(),
            "label_distribution": df["label"].value_counts().to_dict(),
            "dataset_distribution": df["dataset"].value_counts().to_dict(),
        }
        return report
