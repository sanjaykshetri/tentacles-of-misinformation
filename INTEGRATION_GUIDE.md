# Integration Guide - Unified Pipeline & Experiment Tracking

Complete guide to using the new unified data pipeline and experiment tracking infrastructure.

**Status**: ✅ All components operational and tested

## What's New

### 1. **Unified Data Pipeline** (`data/pipeline/`)
- **Purpose**: Reproducible, modular data preparation workflow
- **Components**: Loader → Cleaner → Transformer → Orchestrator
- **Status**: ✅ Pipeline successfully processes 21,724 articles in ~45 seconds

### 2. **Experiment Tracking** (`src/experiment_tracker.py`)
- **Purpose**: Track model experiments with MLflow (or local fallback)
- **Features**: Parameter logging, metric tracking, artifact management
- **Status**: ✅ Local tracking operational, MLflow optional

### 3. **Enhanced Training Scripts**
- **Old**: `src/train_baseline.py` (hardcoded data paths)
- **New**: `src/train_baseline_v2.py` (uses pipeline)
- **Tracked**: `src/train_baseline_tracked.py` (with experiment tracking)

### 4. **Task Runners**
- **`Makefile`**: For macOS/Linux users
- **`tasks.ps1`**: For Windows users (PowerShell)

---

## Quick Start

### Option 1: Windows PowerShell
```powershell
cd tentacles-of-misinformation
.\tasks.ps1 -Task setup          # One-time setup
.\tasks.ps1 -Task train-tracked  # Train with tracking
.\tasks.ps1 -Task status         # Check project status
```

### Option 2: macOS/Linux (Make)
```bash
cd tentacles-of-misinformation
make setup                  # One-time setup
make train-tracked         # Train with tracking
make status                # Check project status
```

### Option 3: Python directly
```bash
cd tentacles-of-misinformation
python data/pipeline/quickstart.py           # Test pipeline
python src/train_baseline_tracked.py         # Train models
```

---

## Detailed Usage

### Running the Pipeline

**Minimal usage** (loads, cleans, engineers features):
```python
from data.pipeline import MisinformationPipeline

pipeline = MisinformationPipeline()
df_processed = pipeline.run(save=True)
print(f"Articles: {len(df_processed)}")
print(f"Features: {df_processed.shape[1]}")
```

**With custom configuration**:
```python
from data.pipeline import PipelineConfig, MisinformationPipeline

config = PipelineConfig(
    min_title_length=10,        # Only keep titles with 10+ chars
    remove_duplicates=True,
    verbose=True,
)

pipeline = MisinformationPipeline(config)
df = pipeline.run(save=True)
```

**Individual stages**:
```python
from data.pipeline import DataLoader, DataCleaner, FeatureTransformer

# Load
loader = DataLoader()
raw_df = loader.load()  # 23,196 articles

# Clean
cleaner = DataCleaner()
clean_df = cleaner.clean(raw_df)  # 21,724 after dedup

# Transform
transformer = FeatureTransformer()
features = transformer.transform(clean_df)  # 13 features
```

### Training with Experiment Tracking

**Basic training**:
```python
from src.experiment_tracker import get_tracker

tracker = get_tracker()

with tracker.run(run_name="my_experiment"):
    tracker.log_params({"lr": 0.01, "epochs": 3})
    
    # ... train model ...
    
    tracker.log_metrics({"accuracy": 0.85, "f1": 0.82})
    tracker.log_artifact(model_path)
```

**Run provided script**:
```bash
python src/train_baseline_tracked.py
```

This script:
1. Loads data via pipeline
2. Splits into train/val
3. Vectorizes with TF-IDF
4. Trains Logistic Regression and Linear SVM
5. Logs all experiments to `experiments/` directory
6. Saves results JSON to `results/`

**Expected output**:
```
Logistic Regression: 81.20% accuracy, 0.8590 ROC-AUC
Linear SVM: 79.36% accuracy, 0.8413 ROC-AUC
```

---

## Project Structure (Updated)

```
tentacles-of-misinformation/
├── data/
│   ├── raw/fakenewsnet/          ← FakeNewsNet CSV files
│   ├── processed/                 ← Pipeline outputs (.parquet)
│   └── pipeline/                  ← NEW: Unified pipeline
│       ├── __init__.py
│       ├── config.py              ← Configuration & constants
│       ├── loader.py              ← Load CSV files
│       ├── cleaner.py             ← Clean & validate
│       ├── transformers.py        ← Feature extraction
│       ├── orchestrator.py        ← Unified workflow
│       ├── quickstart.py          ← Quick-start example
│       └── README.md              ← Pipeline documentation
│
├── src/
│   ├── train_baseline.py          ← OLD: Hardcoded paths
│   ├── train_baseline_v2.py       ← NEW: Pipeline-based
│   ├── train_baseline_tracked.py  ← NEW: With tracking
│   ├── experiment_tracker.py      ← NEW: MLflow integration
│   ├── features.py
│   ├── data_prep.py
│   └── ...
│
├── models/                         ← Trained model artifacts
│   └── logistic_regression_baseline.joblib
│
├── results/                        ← Result summaries
│   └── baseline_summary_*.json
│
├── experiments/                    ← Experiment tracking (local)
│   └── Logistic Regression_*.json
│
├── book/                           ← Quarto book (enhanced with code)
│   ├── chapters/
│   │   ├── 01-measuring-vulnerability.qmd
│   │   ├── 02-detecting-narratives.qmd  ← Added pipeline examples
│   │   └── ...
│   └── _book/
│
├── Makefile                        ← Task runner (Unix/Linux/macOS)
├── tasks.ps1                       ← Task runner (Windows PowerShell)
└── ...
```

---

## Pipeline Data Flow

```
data/raw/fakenewsnet/
  ├── politifact_real.csv (624 articles)
  ├── politifact_fake.csv (432 articles)
  ├── gossipcop_real.csv (16,817 articles)
  └── gossipcop_fake.csv (5,323 articles)
              ↓
          [LOADER]
          23,196 articles loaded
              ↓
          [CLEANER]
          • Remove nulls
          • Remove duplicates (1,472 removed)
          • Filter short titles
              ↓
          21,724 cleaned articles
              ↓
          [TRANSFORMER]
          Extract 13 features:
          • Basic text (2)
          • Sentiment (3)
          • Subjectivity (1)
          • Readability (2)
          • Lexical (1)
          • Language markers (4)
              ↓
      data/processed/
      ├── articles_raw.parquet
      ├── articles_cleaned.parquet
      ├── features.parquet
      └── articles_processed.parquet (ready for ML)
```

---

## Available Tasks

### Windows (PowerShell)
```powershell
.\tasks.ps1 -Task help              # Show all tasks
.\tasks.ps1 -Task setup             # Install + create directories
.\tasks.ps1 -Task pipeline          # Run data pipeline
.\tasks.ps1 -Task train             # Train baseline models
.\tasks.ps1 -Task train-tracked     # Train with tracking
.\tasks.ps1 -Task evaluate          # Show latest results
.\tasks.ps1 -Task book              # Build Quarto book
.\tasks.ps1 -Task book-preview      # Preview book locally
.\tasks.ps1 -Task lint              # Lint Python code
.\tasks.ps1 -Task format            # Format code with black
.\tasks.ps1 -Task test              # Run tests
.\tasks.ps1 -Task status            # Show project status
.\tasks.ps1 -Task clean             # Remove cache files
.\tasks.ps1 -Task clean-all         # Full cleanup
```

### macOS/Linux (Make)
```bash
make help                           # Show all tasks
make setup                          # Install + create directories
make pipeline                       # Run data pipeline
make train                          # Train baseline models
make train-tracked                  # Train with tracking
make evaluate                       # Show latest results
make book                           # Build Quarto book
make book-preview                   # Preview book locally
make lint                           # Lint Python code
make format                         # Format code with black
make test                           # Run tests
make status                         # Show project status
make clean                          # Remove cache files
make clean-all                      # Full cleanup
```

---

## Generated Artifacts

After running the pipeline and training:

### Data Files
- `data/processed/articles_raw.parquet` — Raw loaded data (23,196 rows)
- `data/processed/articles_cleaned.parquet` — After cleaning (21,724 rows)
- `data/processed/features.parquet` — Extracted features (13 columns)
- `data/processed/articles_processed.parquet` — Final dataset (18 columns)

### Models
- `models/tfidf_vectorizer.joblib` — TF-IDF vectorizer
- `models/logistic_regression_baseline.joblib` — Best-performing model
- `models/linear_svm_baseline.joblib` — Comparison model

### Results
- `results/baseline_summary_*.json` — Training summary with metrics
- `experiments/*.json` — Detailed per-model experiment tracking

---

## Configuration

### Pipeline Configuration
Edit `data/pipeline/config.py` to customize:

```python
PipelineConfig(
    raw_data_dir="data/raw/fakenewsnet",
    processed_data_dir="data/processed",
    min_title_length=5,                 # Minimum title chars
    remove_duplicates=True,
    tfidf_params={                      # Vectorization settings
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_df": 0.8,
    },
    verbose=True,
)
```

### Linguistic Markers
Edit `data/pipeline/config.py` constants:

```python
CERTAINTY_TERMS = ["always", "never", "definitely", ...]
HEDGING_TERMS = ["might", "could", "allegedly", ...]
EMOTIONAL_INTENSIFIERS = ["shocking", "amazing", ...]
```

---

## Troubleshooting

### Pipeline fails to find CSV files
```bash
# Check if raw data exists
ls data/raw/fakenewsnet/
# If empty, copy from FakeNewsNet directory
cp FakeNewsNet/dataset/*.csv data/raw/fakenewsnet/
```

### Training out of memory
- Use smaller `max_features` in config: `max_features=5000`
- Or process data in batches

### MLflow not found
- MLflow is optional; local tracking will be used instead
- To install: `pip install mlflow`

### Book rendering fails
- Ensure Quarto is installed: https://quarto.org/docs/get-started/
- Run: `quarto --version`

---

## Integration with Notebooks

All notebooks can now use the pipeline:

```python
# In any notebook
from data.pipeline import MisinformationPipeline

# One-line data loading
df = MisinformationPipeline().run(save=False)

# Proceed with analysis
print(df.head())
print(df.info())
```

---

## Performance Benchmark

**Pipeline Execution** (single run):
- Load: ~2 seconds
- Clean: ~1 second
- Transform: ~30-45 seconds (depends on VADER/textstat availability)
- **Total: ~45 seconds for 21K articles**

**Training** (TF-IDF + 2 models):
- Vectorization: ~10 seconds
- Logistic Regression: ~30 seconds
- Linear SVM: ~20 seconds
- **Total: ~70 seconds**

---

## Next Steps

1. **Add transformer models**: Create `src/train_transformers_v2.py` using DistilBERT
2. **Extend experiment tracking**: Add hyperparameter optimization with Optuna
3. **Build dashboards**: Create Streamlit/Dash dashboard for model monitoring
4. **CI/CD pipeline**: GitHub Actions for automated training on data updates

---

## References

- [Unified Pipeline README](data/pipeline/README.md)
- [Pipeline Configuration](data/pipeline/config.py)
- [Experiment Tracker](src/experiment_tracker.py)
- [Training Script (Tracked)](src/train_baseline_tracked.py)
- [Book "Detecting Narratives" Chapter](book/chapters/02-detecting-narratives.qmd)
