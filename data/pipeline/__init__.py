"""
Unified Data Pipeline for Misinformation Detection

Orchestrates the complete workflow from raw FakeNewsNet data through feature engineering
and model-ready datasets.

Usage:
    from data.pipeline import orchestrator
    pipeline = orchestrator.MisinformationPipeline()
    df_processed = pipeline.run()
"""

from .config import PipelineConfig
from .loader import DataLoader
from .cleaner import DataCleaner
from .transformers import FeatureTransformer
from .orchestrator import MisinformationPipeline

__all__ = [
    "PipelineConfig",
    "DataLoader",
    "DataCleaner",
    "FeatureTransformer",
    "MisinformationPipeline",
]
