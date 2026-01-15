# Data

## Organization

- **raw/** - Original, unmodified datasets
- **processed/** - Cleaned, preprocessed data ready for analysis
- **external/** - Third-party datasets (FakeNewsNet, LIAR, etc.)

## Data Inventory

### Behavioral Data (raw)
- Survey responses (anonymized, IRB-approved)
- Cognitive assessment scores
- Demographic information

### Misinformation Datasets (external)
- FakeNewsNet: News articles with social context
- LIAR: Statements with fact-check verdicts
- Custom misinformation corpus

### Processed Data
- Cleaned survey data with feature engineering
- Tokenized and embedded text
- Train/test splits

## Data Privacy & Ethics

- All behavioral data is anonymized per IRB protocol
- No personally identifiable information in processed files
- External datasets used per their licensing agreements
- GDPR compliant data handling

## Loading Data

```python
import pandas as pd

# Behavioral data
behavioral_data = pd.read_csv('processed/behavioral_features.csv')

# Misinformation texts
news_data = pd.read_csv('processed/news_articles_with_labels.csv')
```

## Note

- Large files (>100MB) are stored in `.gitignore`
- Use Git LFS for model checkpoints
- Raw data files should never be committed
- Only processed, anonymized data in repository

## Contact

For data access or IRB approval questions, contact the author.
