#!/usr/bin/env pwsh
<#
.SYNOPSIS
Task runner for the Misinformation Detection project (Windows-friendly)

.DESCRIPTION
PowerShell-based task runner for common development tasks.

.EXAMPLE
.\tasks.ps1 -Task pipeline
.\tasks.ps1 -Task train-tracked
.\tasks.ps1 -Task help

#>

param(
    [Parameter(Mandatory=$false)]
    [string]$Task = "help"
)

# Project root
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "═" * 70 -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "═" * 70 -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Write-Error2 {
    param([string]$Text)
    Write-Host "✗ $Text" -ForegroundColor Red
}

# Task definitions
$Tasks = @{
    "help" = {
        Write-Host @"
Misinformation Detection Project - Task Runner
==============================================

USAGE: .\tasks.ps1 -Task <task-name>

Setup & Installation:
  install           Install Python dependencies
  setup             Complete project setup (install + dirs + env)

Data Pipeline:
  pipeline          Run unified data pipeline
  data-clean        Clean processed data files

Model Training:
  train             Train baseline models
  train-tracked     Train with MLflow experiment tracking
  evaluate          Evaluate trained models

Documentation:
  book              Build Quarto book locally
  book-preview      Preview book in browser
  book-clean        Clean rendered book files

Code Quality:
  lint              Lint Python files (flake8)
  format            Format code with black
  test              Run tests

Utilities:
  status            Show project status
  clean             Remove cache and temp files
  clean-all         Full cleanup (models, results, etc.)

"@
    }

    "install" = {
        Write-Header "Installing Dependencies"
        pip install -r (Join-Path $ProjectRoot "environment/requirements.txt")
        if ($LASTEXITCODE -eq 0) { Write-Success "Dependencies installed" }
    }

    "setup" = {
        Write-Header "Project Setup"
        & $Tasks["install"].Invoke()
        
        # Create directories
        $dirs = @(
            "data/raw/fakenewsnet",
            "data/processed",
            "models",
            "results",
            "experiments"
        )
        
        foreach ($dir in $dirs) {
            $Path = Join-Path $ProjectRoot $dir
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
            Write-Success "Created directory: $dir"
        }
        
        Write-Success "Project setup complete"
    }

    "pipeline" = {
        Write-Header "Running Data Pipeline"
        $cmd = @"
import sys
sys.path.insert(0, '.')
from data.pipeline import MisinformationPipeline
pipeline = MisinformationPipeline()
df = pipeline.run(save=True)
"@
        python -c $cmd
    }

    "data-clean" = {
        Write-Header "Cleaning Processed Data"
        $ProcessedDir = Join-Path $ProjectRoot "data/processed"
        Get-ChildItem -Path $ProcessedDir -Filter "*.parquet" -ErrorAction SilentlyContinue | Remove-Item -Force
        Write-Success "Cleaned processed data"
    }

    "train" = {
        Write-Header "Training Baseline Models"
        & $Tasks["pipeline"].Invoke()
        python (Join-Path $ProjectRoot "src/train_baseline_v2.py")
    }

    "train-tracked" = {
        Write-Header "Training with Experiment Tracking"
        & $Tasks["pipeline"].Invoke()
        Write-Host "Installing MLflow (optional)..."
        pip install mlflow -q 2>$null
        python (Join-Path $ProjectRoot "src/train_baseline_tracked.py")
    }

    "evaluate" = {
        Write-Header "Evaluating Models"
        $cmd = @"
import json
from pathlib import Path
results = sorted(Path('results').glob('*summary*.json'), key=lambda x: x.stat().st_mtime)
if results:
    with open(results[-1]) as f:
        print(json.dumps(json.load(f), indent=2))
else:
    print('No results found')
"@
        python -c $cmd
    }

    "book" = {
        Write-Header "Building Quarto Book"
        Set-Location (Join-Path $ProjectRoot "book")
        quarto render
        Set-Location $ProjectRoot
        Write-Success "Book built to: book/_book/index.html"
    }

    "book-preview" = {
        Write-Header "Starting Book Preview"
        Write-Host "Press Ctrl+C to stop..."
        Set-Location (Join-Path $ProjectRoot "book")
        quarto preview
        Set-Location $ProjectRoot
    }

    "book-clean" = {
        Write-Header "Cleaning Book Artifacts"
        $dirs = @("_book", ".quarto")
        foreach ($dir in $dirs) {
            $Path = Join-Path (Join-Path $ProjectRoot "book") $dir
            if (Test-Path $Path) {
                Remove-Item -Path $Path -Recurse -Force
                Write-Success "Removed: $dir"
            }
        }
    }

    "lint" = {
        Write-Header "Linting Python Code"
        if (-not (Get-Command flake8 -ErrorAction SilentlyContinue)) {
            Write-Host "Installing flake8..."
            pip install flake8 -q
        }
        flake8 (Join-Path $ProjectRoot "src") (Join-Path $ProjectRoot "data/pipeline") --max-line-length=100
    }

    "format" = {
        Write-Header "Formatting Python Code"
        if (-not (Get-Command black -ErrorAction SilentlyContinue)) {
            Write-Host "Installing black..."
            pip install black -q
        }
        black (Join-Path $ProjectRoot "src") (Join-Path $ProjectRoot "data/pipeline") --line-length=100
        Write-Success "Code formatted"
    }

    "test" = {
        Write-Header "Running Tests"
        if (-not (Get-Command pytest -ErrorAction SilentlyContinue)) {
            Write-Host "Installing pytest..."
            pip install pytest -q
        }
        pytest (Join-Path $ProjectRoot "tests") -v
    }

    "status" = {
        Write-Header "Project Status"
        
        Write-Host "Data:" -ForegroundColor Yellow
        $rawCount = (Get-ChildItem (Join-Path $ProjectRoot "data/raw/fakenewsnet") -Filter "*.csv" -ErrorAction SilentlyContinue).Count
        Write-Host "  Raw: $rawCount files"
        
        $procCount = (Get-ChildItem (Join-Path $ProjectRoot "data/processed") -Filter "*.parquet" -ErrorAction SilentlyContinue).Count
        Write-Host "  Processed: $procCount files"
        
        Write-Host ""
        Write-Host "Models:" -ForegroundColor Yellow
        $modelCount = (Get-ChildItem (Join-Path $ProjectRoot "models") -Filter "*.joblib" -ErrorAction SilentlyContinue).Count
        if ($modelCount -gt 0) { Write-Host "  $modelCount model(s)" } else { Write-Host "  No models trained" }
        
        Write-Host ""
        Write-Host "Results:" -ForegroundColor Yellow
        $resultCount = (Get-ChildItem (Join-Path $ProjectRoot "results") -Filter "*.json" -ErrorAction SilentlyContinue).Count
        Write-Host "  $resultCount result(s)"
        
        Write-Host ""
        Write-Host "Book:" -ForegroundColor Yellow
        $bookPath = Join-Path (Join-Path $ProjectRoot "book") "_book/index.html"
        if (Test-Path $bookPath) { Write-Host "  ✓ Built" } else { Write-Host "  ✗ Not built" }
    }

    "clean" = {
        Write-Header "Cleaning Cache & Temp Files"
        $patterns = @("__pycache__", "*.pyc", ".pytest_cache", ".DS_Store")
        foreach ($pattern in $patterns) {
            Get-ChildItem -Path $ProjectRoot -Recurse -ErrorAction SilentlyContinue | 
                Where-Object { $_.Name -like $pattern } | 
                Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Success "Cleaned"
    }

    "clean-all" = {
        Write-Header "Full Project Cleanup"
        & $Tasks["clean"].Invoke()
        & $Tasks["book-clean"].Invoke()
        & $Tasks["data-clean"].Invoke()
        
        $dirs = @("models", "results", "experiments", "mlruns")
        foreach ($dir in $dirs) {
            $Path = Join-Path $ProjectRoot $dir
            Get-ChildItem -Path $Path -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Success "Full cleanup complete"
    }
}

# Execute task
if ($Tasks.ContainsKey($Task)) {
    & $Tasks[$Task]
} else {
    Write-Error2 "Unknown task: $Task"
    Write-Host "Run '.\tasks.ps1 -Task help' for available tasks"
    exit 1
}
