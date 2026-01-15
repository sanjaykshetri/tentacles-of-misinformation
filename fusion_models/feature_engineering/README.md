# Fusion Models — Feature Engineering

## Purpose

Extract, engineer, and prepare features from both behavioral and content modalities for use in multimodal models.

## Behavioral Features

### Cognitive Assessments
- Cognitive Reflection Test (CRT) scores
- Numeracy assessments
- Need for Cognition (NFC)
- Belief in Conspiracy Theories

### Personality & Mindset
- Big Five personality traits
- Political polarization
- Trust in institutions
- Openness to counter-arguments

### Demographics
- Age, education, income
- Political affiliation
- Media consumption habits
- Social media usage frequency

### Processing

```python
from feature_engineering import behavioral_preprocessor

features = behavioral_preprocessor.extract_and_normalize(survey_data)
# Returns: standardized behavioral feature matrix
```

## Content Features

### Text Embeddings
- SBERT embeddings (384-dim)
- Contextual representations

### Misinformation Scores
- Classification confidence from NLP model
- Predicted label probability

### Linguistic Features
- Readability metrics (Flesch-Kincaid)
- Sentiment polarity and subjectivity
- Named entity density

### Processing

```python
from feature_engineering import content_preprocessor

embeddings = content_preprocessor.get_sbert_embeddings(texts)
misinfo_scores = content_preprocessor.get_classification_scores(texts)
```

## Feature Selection

- Correlation analysis
- Feature importance from baseline models
- Dimensionality reduction (PCA)
- Domain expert review

## Output

Feature matrices ready for multimodal fusion architectures.
