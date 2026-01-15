# Fusion Models

## Overview

This module integrates behavioral features (cognitive vulnerability, demographics) with NLP features (text embeddings, misinformation scores) into multimodal models for improved risk prediction.

## Architecture

**Input Modalities:**
- Behavioral: cognitive assessment scores, personality traits, demographics
- Content: text embeddings (SBERT), misinformation classification outputs

**Output:**
- Individual-level misinformation susceptibility risk scores
- Content-level misinformation severity assessments

## Components

- **feature_engineering/** - Feature extraction, normalization, and selection
- **multimodal_models/** - Fusion architectures (early fusion, late fusion, attention-based)
- **experiments/** - Experimental runs, hyperparameter tuning, ablation studies

## Fusion Approaches

1. **Early Fusion**: Concatenate features before neural network
2. **Late Fusion**: Separate networks for each modality, then combine
3. **Attention Fusion**: Cross-modal attention mechanisms
4. **Ensemble Fusion**: Weighted predictions from modality-specific models

## Key Metrics

- ROC-AUC of susceptibility predictions
- Feature importance (SHAP values)
- Cross-modal correlation analysis
- Ablation study results

## Technologies

- PyTorch for neural networks
- scikit-learn for preprocessing
- SHAP for explainability
- optuna for hyperparameter optimization

## Getting Started

```python
from fusion_models import feature_engineering, multimodal_models

# Prepare behavioral and content features
behavioral_features = feature_engineering.extract_behavioral(survey_data)
content_features = feature_engineering.extract_content(nlp_embeddings)

# Train multimodal model
model = multimodal_models.FusionModel(fusion_type='attention')
model.fit(behavioral_features, content_features, labels)

# Predict susceptibility
risk_scores = model.predict(new_behavioral, new_content)
```

## Results Summary

- Model performance improvements over single-modality baselines
- Feature importance rankings
- Predictive insights for policy recommendations

## References

- Fusion architectures in multimodal learning literature
- Related work on behavioral + content modeling
