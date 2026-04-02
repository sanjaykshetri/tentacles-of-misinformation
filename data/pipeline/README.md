# Data Pipeline Module

A unified, production-grade data pipeline for the misinformation detection project.

**Status**: ✨ Unified pipeline operational (v1.0)

## Quick Start

Run the complete pipeline in one command:

```python
from data.pipeline import MisinformationPipeline

# Initialize and run
pipeline = MisinformationPipeline()
df_processed = pipeline.run(save=True)

print(f"Processed articles: {len(df_processed)}")
print(f"Features extracted: {df_processed.shape[1]}")
```

## Pipeline Architecture

The pipeline orchestrates four modular stages:

```
Raw Data (FakeNewsNet CSVs)
    ↓
[LOADER] Load & combine CSV files
    ↓
[CLEANER] Remove nulls, duplicates, short titles
    ↓
[TRANSFORMER] Extract linguistic features
    ↓
Processed Data (Parquet with all features)
```

### Stage 1: Loader

Loads all FakeNewsNet CSV files from `data/raw/fakenewsnet/`:

```python
from data.pipeline import DataLoader, PipelineConfig

config = PipelineConfig()
loader = DataLoader(config)
df_raw = loader.load()
```

**Output**: Combined dataframe with columns:
- `id`: Article ID
- `title`: Article title (text)
- `url`: Source URL
- `label`: Real/Fake (binary label)
- `dataset`: Source dataset (politifact/gossipcop)

### Stage 2: Cleaner

Validates and cleans data:

```python
from data.pipeline import DataCleaner

cleaner = DataCleaner(config)
df_clean = cleaner.clean(df_raw)

# Get quality metrics
quality = cleaner.get_quality_report(df_clean)
print(f"Retention rate: {quality['total_articles']/len(df_raw)*100:.1f}%")
```

**Cleaning steps**:
1. Remove missing titles
2. Strip whitespace
3. Remove empty strings
4. Filter short titles (< 5 chars)
5. Deduplicate by title

### Stage 3: Transformer

Extracts 11+ linguistic features:

```python
from data.pipeline import FeatureTransformer

transformer = FeatureTransformer(config)
features = transformer.transform(df_clean, text_column="title")
```

**Features extracted**:

| Category | Features |
|----------|----------|
| **Text Basics** | title_length_words, title_length_chars |
| **Sentiment** | sentiment_compound, sentiment_positive, sentiment_negative |
| **Subjectivity** | subjectivity (0=objective, 1=subjective) |
| **Readability** | flesch_kincaid_grade, ari |
| **Lexical** | lexical_diversity (type-token ratio) |
| **Language** | certainty_terms, hedging_terms, emotional_intensifiers, certainty_hedging_ratio |

### Stage 4: Orchestrator

Runs the complete pipeline end-to-end:

```python
from data.pipeline import MisinformationPipeline

pipeline = MisinformationPipeline()
df_final = pipeline.run(save=True)

# Get statistics
stats = pipeline.get_statistics()
print(f"Raw size: {stats['raw_size']}")
print(f"Cleaned size: {stats['cleaned_size']}")
print(f"Removal rate: {stats['removal_rate']:.1f}%")
print(f"Features: {stats['features_extracted']}")
```

**Saved outputs**:
- `data/processed/articles_raw.parquet` - Raw loaded data
- `data/processed/articles_cleaned.parquet` - After cleaning
- `data/processed/features.parquet` - Extracted features only
- `data/processed/articles_processed.parquet` - Final dataset (complete)

## Configuration

Customize pipeline behavior via `PipelineConfig`:

```python
from data.pipeline import PipelineConfig

config = PipelineConfig(
    raw_data_dir="data/raw/fakenewsnet",      # Where to load CSVs
    processed_data_dir="data/processed",       # Where to save outputs
    models_dir="models",
    results_dir="results",
    min_title_length=5,                        # Minimum title length
    remove_duplicates=True,
    verbose=True,                              # Print logging
)

pipeline = MisinformationPipeline(config)
df = pipeline.run(save=True)
```

## Advanced Usage

### Run individual stages

```python
from data.pipeline import DataLoader, DataCleaner, FeatureTransformer

config = PipelineConfig()

# Manual orchestration
loader = DataLoader(config)
raw_df = loader.load()

cleaner = DataCleaner(config)
clean_df = cleaner.clean(raw_df)

transformer = FeatureTransformer(config)
features = transformer.transform(clean_df)

# Combine manually
import pandas as pd
result = pd.concat([clean_df, features], axis=1)
```

### Access linguistic markers

Import term lists for custom analysis:

```python
from data.pipeline.config import CERTAINTY_TERMS, HEDGING_TERMS

print(f"Certainty markers: {CERTAINTY_TERMS}")
print(f"Hedging markers: {HEDGING_TERMS}")
```

### Feature statistics by label

```python
import pandas as pd

# Calculate mean features by label
feature_stats = df_processed.groupby('label')[[
    'sentiment_compound',
    'subjectivity',
    'flesch_kincaid_grade',
    'certainty_terms',
    'hedging_terms'
]].mean()

print(feature_stats.round(3))
```

## Dependencies

The pipeline handles missing dependencies gracefully:

```
Required:
  - pandas
  - numpy

Optional (auto-detected):
  - vaderSentiment (sentiment analysis)
  - textblob (subjectivity)
  - textstat (readability)
```

Install all optional dependencies:

```bash
pip install vaderSentiment textblob textstat
```

## Data Quality Metrics

The pipeline generates quality metrics after cleaning:

```python
quality = cleaner.get_quality_report(df_clean)
# Returns:
# - total_articles
# - missing_titles
# - duplicate_titles
# - avg_title_length
# - min/max_title_length
# - label_distribution (dict)
# - dataset_distribution (dict)
```

## Integration with Models

Use the pipeline output for training:

```python
from data.pipeline import MisinformationPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Get processed data
pipeline = MisinformationPipeline()
df = pipeline.run(save=False)

# Vectorize
vec = TfidfVectorizer(**config.tfidf_params)
X = vec.fit_transform(df['title'])
y = (df['label'] == 'fake').astype(int)

# Train
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)
```

## Debugging

### Enable verbose logging

```python
config = PipelineConfig(verbose=True)
pipeline = MisinformationPipeline(config)
df = pipeline.run()
```

### Check file existence

```python
from pathlib import Path
from data.pipeline import PipelineConfig

config = PipelineConfig()
print(f"Raw data dir exists: {config.raw_data_dir.exists()}")
print(f"Files in raw dir: {list(config.raw_data_dir.glob('*.csv'))}")
```

## Performance

Typical runtime on standard hardware:
- **Load**: ~2-3 seconds
- **Clean**: ~1 second
- **Transform**: ~30-45 seconds (depends on VADER/textstat availability)
- **Total**: ~35-50 seconds for ~21K articles

## Reproducibility

All operations use:
- Fixed random seed (42)
- Deterministic feature extraction
- Logged parameters in `config`
- Versioned dependencies in `environment/requirements.txt`

## Contributing

To extend the pipeline:

1. Add new feature in `data/pipeline/transformers.py`
2. Update `config.py` with parameters
3. Add to `orchestrator.py` pipeline
4. Document in this README

## References

- [Unified Pipeline Architecture](../REPO_STRUCTURE.md)
- [Example Notebooks](../behavioral_analysis/notebooks/)
- [FakeNewsNet Dataset](../FakeNewsNet/README.md)
