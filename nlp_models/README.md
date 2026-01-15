# NLP Models

## Overview

This module builds natural language processing pipelines to detect misinformation narratives and classify text at scale using transformer models and classical NLP techniques.

## Components

- **preprocessing/** - Text cleaning, tokenization, and normalization
- **embeddings/** - SBERT, transformer-based embeddings, and representation learning
- **classification/** - Misinformation classifiers (BERT, RoBERTa, DistilBERT)
- **evaluation/** - Metrics, cross-validation, and performance analysis

## Key Features

- Multi-label classification for misinformation types
- Transfer learning from pre-trained transformers
- Embeddings for semantic similarity search
- Ensemble methods for robust predictions

## Datasets

- FakeNewsNet
- LIAR dataset
- Custom misinformation corpus
- Public tweets and articles

## Technologies

- transformers (Hugging Face)
- torch / PyTorch
- scikit-learn
- nltk, spacy

## Getting Started

```python
# Load and preprocess text data
from nlp_models.preprocessing import cleaner
from nlp_models.classification import classifier

cleaned_texts = cleaner.preprocess(raw_texts)
predictions = classifier.predict_misinformation(cleaned_texts)
```

## Model Checkpoints

- Trained models stored in `models/` (Git LFS)
- Baseline models available from Hugging Face Hub

## Evaluation Metrics

- Precision, Recall, F1-Score
- ROC-AUC for ranking
- Confusion matrices by misinformation type

## References

- Devlin et al. (2018) - BERT
- Pennycook & Rand (2021) - Misinformation psychology
