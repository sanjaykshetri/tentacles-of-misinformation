#!/usr/bin/env python
"""
Quick-start example for the unified data pipeline.

Usage:
    python data/pipeline/examples/quickstart.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline import MisinformationPipeline, PipelineConfig


def main():
    """Run the complete pipeline with examples."""
    
    print("\n" + "="*70)
    print("MISINFORMATION DETECTION PIPELINE - QUICK START")
    print("="*70)
    
    # Initialize pipeline with default configuration
    config = PipelineConfig(verbose=True)
    
    print("\n📋 Pipeline Configuration:")
    print(f"  • Raw data dir: {config.raw_data_dir}")
    print(f"  • Output dir: {config.processed_data_dir}")
    print(f"  • Min title length: {config.min_title_length}")
    print(f"  • Remove duplicates: {config.remove_duplicates}")
    
    try:
        # Run the pipeline
        pipeline = MisinformationPipeline(config)
        df_processed = pipeline.run(save=True)
        
        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        # Get statistics
        stats = pipeline.get_statistics()
        print(f"\n📊 Pipeline Statistics:")
        print(f"  • Raw articles: {stats['raw_size']}")
        print(f"  • Cleaned articles: {stats['cleaned_size']}")
        print(f"  • Removal rate: {stats['removal_rate']:.1f}%")
        print(f"  • Features extracted: {stats['features_extracted']}")
        
        # Dataset info
        print(f"\n📈 Final Dataset:")
        print(f"  • Shape: {df_processed.shape}")
        print(f"  • Columns: {list(df_processed.columns)}")
        
        # Label distribution
        print(f"\n📌 Label Distribution:")
        for label, count in df_processed['label'].value_counts().items():
            pct = (count / len(df_processed)) * 100
            print(f"  • {label}: {count} ({pct:.1f}%)")
        
        # Feature statistics
        print(f"\n📝 Feature Statistics by Label:")
        feature_cols = [
            'sentiment_compound', 'subjectivity', 'certainty_terms',
            'hedging_terms', 'emotional_intensifiers'
        ]
        feature_stats = df_processed.groupby('label')[feature_cols].mean()
        print(feature_stats.round(3))
        
        # Sample articles
        print(f"\n🔍 Sample Articles (first 3):")
        for idx, row in df_processed.head(3).iterrows():
            print(f"\n  [{row['label'].upper()}] {row['title'][:60]}...")
            print(f"    Sentiment: {row['sentiment_compound']:.2f} | "
                  f"Subjectivity: {row['subjectivity']:.2f} | "
                  f"Certainty: {row['certainty_terms']}")
        
        # Success message
        print("\n" + "✅"*35)
        print("PIPELINE EXECUTION COMPLETE!")
        print("✅"*35)
        print(f"\n✨ All outputs saved to: {config.processed_data_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
