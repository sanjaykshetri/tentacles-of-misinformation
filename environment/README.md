# Environment Setup

## Python Environment

Two ways to set up your development environment:

### Option 1: Conda (Recommended)

```bash
# Create environment
conda env create -f conda.yml

# Activate environment
conda activate tentacles-misinformation

# Verify installation
python --version
```

### Option 2: Pip + Venv

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## IDE Setup

### VS Code
- Install Python extension
- Select interpreter from created environment
- Install Jupyter extension for notebooks

### JupyterLab
```bash
jupyter lab
```

## Environment Variables

Create `.env` file in project root:
```
DATA_PATH=./data
MODEL_PATH=./models
LOG_LEVEL=INFO
```

## Verification

```python
# Test imports
python -c "import pandas, torch, transformers; print('✓ All imports successful')"
```

## Updates

To update packages:
```bash
pip install --upgrade -r requirements.txt
# or
conda update --all -n tentacles-misinformation
```

## Troubleshooting

- CUDA issues: Ensure PyTorch CUDA version matches your GPU
- Package conflicts: Consider creating fresh environment
- Missing dependencies: Run `pip install -r requirements.txt` again

## Next Steps

Once environment is ready:
1. Run behavioral analysis notebooks
2. Train NLP baseline models
3. Start fusion model experiments
4. Build dashboard prototypes
