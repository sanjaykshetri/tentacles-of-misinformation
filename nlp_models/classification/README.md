# NLP Models — Classification

## Purpose

Build and evaluate classifiers to detect misinformation narratives and categorize text by credibility and narrative type.

## Tasks

### Binary Classification
- Misinformation vs. Credible

### Multi-class Classification
- Satire
- Misleading
- False
- Credible

### Multi-label Classification
- Multiple misinformation types per article

## Models

### Pre-trained Transformers
- BERT (base, large)
- RoBERTa
- DistilBERT (faster, smaller)
- ALBERT

### Ensemble Methods
- Voting classifier
- Stacking classifier
- Gradient boosting on transformer outputs

## Training

```python
from classification import trainer

model = trainer.train_classifier(
    train_data=df_train,
    text_col='text',
    label_col='label',
    model_type='roberta-base',
    epochs=3,
    batch_size=16
)

# Save trained model
model.save('models/roberta_misinformation_classifier')
```

## Evaluation

- Holdout test set
- Cross-validation (5-fold)
- F1-score, Precision, Recall
- Per-class performance breakdown
- Confusion matrices

## Class Imbalance

- Weighted loss functions
- Data augmentation
- Focal loss
- Class balancing strategies

## Inference

```python
from classification import predictor

predictions = predictor.predict(test_texts, model_path='models/...')
# Returns: predicted labels and confidence scores
```

## References

- Devlin et al. (2018) - BERT
- Liu et al. (2019) - RoBERTa
