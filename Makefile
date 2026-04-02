# Makefile for Misinformation Detection Project
# Usage: make [target]

.PHONY: help install setup pipeline train train-tracked evaluate book test lint format clean

# Default target
help:
	@echo "Misinformation Detection Project - Available Tasks"
	@echo "=================================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install Python dependencies"
	@echo "  setup            Complete project setup (install + data + env)"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  pipeline         Run unified data pipeline"
	@echo "  data-clean       Clean processed data files"
	@echo ""
	@echo "Model Training:"
	@echo "  train            Train baseline models"
	@echo "  train-tracked    Train with MLflow experiment tracking"
	@echo "  evaluate         Evaluate trained models"
	@echo ""
	@echo "Documentation & Rendering:"
	@echo "  book             Build Quarto book locally"
	@echo "  book-preview     Preview book in browser"
	@echo "  book-clean       Clean rendered book files"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Lint Python files (flake8)"
	@echo "  format           Format code with black"
	@echo "  test             Run tests"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Remove cached files and artifacts"
	@echo "  clean-all        Full cleanup (models, results, cache)"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r environment/requirements.txt

# Complete setup
setup: install
	@echo "Setting up project..."
	@mkdir -p data/raw/fakenewsnet data/processed
	@mkdir -p models results experiments
	python -c "from data.pipeline import PipelineConfig; PipelineConfig(verbose=True)"
	@echo "✓ Project setup complete"

# Run unified pipeline
pipeline:
	@echo "Running data pipeline..."
	python -c "import sys; sys.path.insert(0, '.'); from data.pipeline import MisinformationPipeline; pipeline = MisinformationPipeline(); df = pipeline.run(save=True)"

# Clean processed data
data-clean:
	@echo "Cleaning processed data..."
	rm -f data/processed/*.parquet
	@echo "✓ Cleaned"

# Train baseline models
train: pipeline
	@echo "Training baseline models..."
	python src/train_baseline_v2.py

# Train with tracking
train-tracked: pipeline
	@echo "Training models with experiment tracking..."
	pip install mlflow -q 2>/dev/null || echo "Note: MLflow not available, using local tracking"
	python src/train_baseline_tracked.py

# Evaluate models
evaluate:
	@echo "Evaluating trained models..."
	python -c "import json; from pathlib import Path; results = list(Path('results').glob('*summary*.json')); results and print(json.dumps(json.load(open(results[-1])), indent=2)) or print('No results found')"

# Build book
book:
	@echo "Building Quarto book..."
	cd book && quarto render
	@echo "✓ Book built to: book/_book/index.html"

# Preview book
book-preview:
	@echo "Starting book preview (Ctrl+C to stop)..."
	cd book && quarto preview

# Clean book
book-clean:
	@echo "Cleaning book artifacts..."
	rm -rf book/_book book/.quarto
	@echo "✓ Cleaned"

# Lint code
lint:
	@echo "Linting Python code..."
	@which flake8 > /dev/null || pip install flake8 > /dev/null
	flake8 src/ data/pipeline/ --max-line-length=100 --ignore=E501,W503

# Format code
format:
	@echo "Formatting Python code with black..."
	@which black > /dev/null || pip install black > /dev/null
	black src/ data/pipeline/ --line-length=100

# Run tests
test:
	@echo "Running tests..."
	@which pytest > /dev/null || pip install pytest > /dev/null
	pytest tests/ -v

# Clean cache
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f .DS_Store
	@echo "✓ Cleaned"

# Full cleanup
clean-all: clean book-clean data-clean
	@echo "Full cleanup: removing models and results..."
	rm -rf models/*.joblib
	rm -rf results/*.json
	rm -rf experiments/
	rm -rf mlruns/
	@echo "✓ Full cleanup complete"

# Project status
status:
	@echo "Project Status"
	@echo "=============="
	@echo ""
	@echo "Data:"
	@echo -n "  Raw: "
	@ls data/raw/fakenewsnet/*.csv 2>/dev/null | wc -l || echo "0 files"
	@echo -n "  Processed: "
	@ls data/processed/*.parquet 2>/dev/null | wc -l || echo "0 files"
	@echo ""
	@echo "Models:"
	@echo -n "  " && ls models/*.joblib 2>/dev/null || echo "No models trained"
	@echo ""
	@echo "Results:"
	@echo -n "  " && ls results/*.json 2>/dev/null | wc -l || echo "0 results"
	@echo ""
	@echo "Book:"
	@test -f book/_book/index.html && echo "  ✓ Built (book/_book/index.html)" || echo "  ✗ Not built"
