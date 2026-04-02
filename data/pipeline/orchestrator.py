"""
Pipeline Orchestrator

Coordinates the end-to-end workflow from raw data to processed features.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from .config import PipelineConfig
from .loader import DataLoader
from .cleaner import DataCleaner
from .transformers import FeatureTransformer


class MisinformationPipeline:
    """
    Unified pipeline for misinformation detection.
    
    Orchestrates: Load → Clean → Feature Engineering → Save
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        config : PipelineConfig, optional
            Pipeline configuration. If None, uses defaults.
        """
        self.config = config or PipelineConfig()
        self.loader = DataLoader(self.config)
        self.cleaner = DataCleaner(self.config)
        self.transformer = FeatureTransformer(self.config)
        
        # Pipeline state
        self.raw_data = None
        self.cleaned_data = None
        self.features = None
    
    def run(self, save: bool = True) -> pd.DataFrame:
        """
        Execute the full pipeline.
        
        Parameters
        ----------
        save : bool
            Whether to save intermediate outputs to parquet
        
        Returns
        -------
        pd.DataFrame
            Final processed dataset with features
        """
        self.config.log("\n" + "🔷" * 35)
        self.config.log("MISINFORMATION DETECTION PIPELINE")
        self.config.log("🔷" * 35)
        
        # Step 1: Load
        self.raw_data = self.loader.load()
        
        # Step 2: Clean
        self.cleaned_data = self.cleaner.clean(self.raw_data)
        
        # Step 3: Transform
        self.features = self.transformer.transform(self.cleaned_data, text_column="title")
        
        # Step 4: Combine
        df_final = self.cleaned_data[["id", "title", "url", "label", "dataset"]].copy()
        df_final = pd.concat([df_final, self.features], axis=1)
        
        # Step 5: Save (optional)
        if save:
            self._save_outputs(df_final)
        
        self.config.log("\n" + "✨" * 35)
        self.config.log("PIPELINE COMPLETE")
        self.config.log("✨" * 35 + "\n")
        
        return df_final
    
    def _save_outputs(self, df: pd.DataFrame):
        """Save pipeline outputs to disk."""
        self.config.log("\n" + "=" * 70)
        self.config.log("SAVING OUTPUTS")
        self.config.log("=" * 70)
        
        # Save raw data
        raw_path = self.config.processed_data_dir / "articles_raw.parquet"
        self.raw_data.to_parquet(raw_path, index=False)
        self.config.log(f"✓ Raw data: {raw_path}")
        
        # Save cleaned data
        cleaned_path = self.config.processed_data_dir / "articles_cleaned.parquet"
        self.cleaned_data.to_parquet(cleaned_path, index=False)
        self.config.log(f"✓ Cleaned data: {cleaned_path}")
        
        # Save features
        features_path = self.config.processed_data_dir / "features.parquet"
        self.features.to_parquet(features_path, index=False)
        self.config.log(f"✓ Features: {features_path}")
        
        # Save final processed data
        final_path = self.config.processed_data_dir / "articles_processed.parquet"
        df.to_parquet(final_path, index=False)
        self.config.log(f"✓ Processed data: {final_path}")
        
        # Generate summary
        self.config.log(f"\n  Final dataset shape: {df.shape}")
        self.config.log(f"  Columns: {list(df.columns)}\n")
    
    def get_statistics(self) -> dict:
        """Get pipeline execution statistics."""
        stats = {
            "raw_size": len(self.raw_data) if self.raw_data is not None else 0,
            "cleaned_size": len(self.cleaned_data) if self.cleaned_data is not None else 0,
            "features_extracted": self.features.shape[1] if self.features is not None else 0,
        }
        if self.raw_data is not None and self.cleaned_data is not None:
            stats["removal_rate"] = (
                (len(self.raw_data) - len(self.cleaned_data)) / len(self.raw_data) * 100
            )
        return stats


def run_pipeline(
    config: Optional[PipelineConfig] = None,
    save: bool = True
) -> Tuple[pd.DataFrame, MisinformationPipeline]:
    """
    Convenience function to run the pipeline.
    
    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration
    save : bool
        Whether to save outputs
    
    Returns
    -------
    tuple
        (processed_dataframe, pipeline_instance)
    """
    pipeline = MisinformationPipeline(config)
    df = pipeline.run(save=save)
    return df, pipeline


if __name__ == "__main__":
    # Run the pipeline
    df, pipeline = run_pipeline()
    
    # Print statistics
    stats = pipeline.get_statistics()
    print("\n" + "=" * 70)
    print("PIPELINE STATISTICS")
    print("=" * 70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value}")
