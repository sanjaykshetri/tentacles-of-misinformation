"""
Data Loader Module

Loads raw FakeNewsNet CSV files and creates a unified dataframe.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from .config import PipelineConfig


class DataLoader:
    """Load and combine FakeNewsNet CSV files."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        config : PipelineConfig, optional
            Pipeline configuration. If None, uses defaults.
        """
        self.config = config or PipelineConfig()
    
    def load(self) -> pd.DataFrame:
        """
        Load all FakeNewsNet CSV files and combine them.
        
        Returns
        -------
        pd.DataFrame
            Combined dataframe with columns: id, title, url, label, dataset
        """
        self.config.log("=" * 70)
        self.config.log("LOADING RAW DATA")
        self.config.log("=" * 70)
        
        rows = []
        
        for csv_file, (label, source) in self.config.csv_files.items():
            path = self.config.raw_data_dir / csv_file
            
            if path.exists():
                self.config.log(f"✓ Loading {csv_file}...")
                df_temp = pd.read_csv(path)
                
                # Standardize columns
                df_temp["label"] = label
                df_temp["dataset"] = source
                
                # Extract relevant columns
                columns_needed = ["id", "title", "label", "dataset"]
                if "news_url" in df_temp.columns:
                    df_temp.rename(columns={"news_url": "url"}, inplace=True)
                    columns_needed.insert(2, "url")
                elif "url" not in df_temp.columns:
                    df_temp["url"] = ""
                    columns_needed.insert(2, "url")
                
                df_temp = df_temp[columns_needed].copy()
                
                rows.append(df_temp)
                self.config.log(f"  → Loaded {len(df_temp)} articles")
            else:
                self.config.log(f"✗ File not found: {path}")
        
        if not rows:
            raise ValueError(f"No CSV files found in {self.config.raw_data_dir}")
        
        df = pd.concat(rows, ignore_index=True)
        self.config.log(f"\n✅ Total articles loaded: {len(df)}")
        self.config.log(f"   Label distribution:\n{df['label'].value_counts()}\n")
        
        return df
