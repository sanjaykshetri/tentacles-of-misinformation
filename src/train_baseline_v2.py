"""
Baseline NLP Models - Updated to use Unified Pipeline

Trains TF-IDF + Logistic Regression and Linear SVM with class balancing.
Uses the unified data pipeline for reproducible data preparation.

Usage:
    python src/train_baseline_v2.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve, accuracy_score, f1_score
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline import MisinformationPipeline, PipelineConfig

# Paths
SCRIPT_DIR = PROJECT_ROOT
MODEL_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_data_from_pipeline():
    """Load and prepare data using the unified pipeline."""
    print("\n" + "="*70)
    print("LOADING DATA FROM UNIFIED PIPELINE")
    print("="*70)
    
    config = PipelineConfig(verbose=False)
    pipeline = MisinformationPipeline(config)
    
    # Check if processed data already exists
    processed_path = config.processed_data_dir / "articles_processed.parquet"
    
    if processed_path.exists():
        print(f"✓ Loading cached processed data from: {processed_path}")
        df = pd.read_parquet(processed_path)
    else:
        print("Running pipeline to generate processed data...")
        df = pipeline.run(save=True)
    
    # Prepare labels
    df["label_num"] = (df["label"] == "fake").astype(int)
    
    print(f"\n✓ Loaded {len(df)} articles")
    print(f"  Label distribution:\n{df['label'].value_counts()}")
    
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Stratified train/validation split."""
    print(f"\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    
    X_train, X_val, y_train, y_val = train_test_split(
        df["title"],
        df["label_num"],
        test_size=test_size,
        stratify=df["label_num"],
        random_state=random_state
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")
    print(f"Train class distribution:\n{pd.Series(y_train).value_counts()}")
    print(f"Val class distribution:\n{pd.Series(y_val).value_counts()}")
    
    return X_train, X_val, y_train, y_val


def vectorize(X_train, X_val):
    """TF-IDF vectorization with standard parameters."""
    print(f"\n" + "="*70)
    print("VECTORIZING WITH TF-IDF")
    print("="*70)
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    print(f"\n✓ Vectorizer fitted on {len(vectorizer.vocabulary_)} features")
    print(f"  Train matrix shape: {X_train_vec.shape}")
    print(f"  Val matrix shape: {X_val_vec.shape}")
    
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  ✓ Saved to: {vectorizer_path}")
    
    return X_train_vec, X_val_vec, vectorizer


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with balanced class weights."""
    print(f"\n" + "="*70)
    print("TRAINING: LOGISTIC REGRESSION")
    print("="*70)
    
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    
    return model


def train_linear_svm(X_train, y_train):
    """Train Linear SVM with balanced class weights."""
    print(f"\n" + "="*70)
    print("TRAINING: LINEAR SVM")
    print("="*70)
    
    model = LinearSVC(
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    
    return model


def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model on validation set."""
    print(f"\n" + "-"*70)
    print(f"EVALUATING: {model_name}")
    print("-"*70)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.decision_function(X_val)
    
    # Normalize decision function to [0, 1] for ROC-AUC
    y_pred_proba_normalized = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
    
    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba_normalized)
    
    print(f"\nMetrics:")
    print(f"  • Accuracy: {accuracy:.4f}")
    print(f"  • F1 Score: {f1:.4f}")
    print(f"  • ROC-AUC: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Real", "Fake"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]} | TP: {cm[1,1]}")
    
    results = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "trained_at": datetime.now().isoformat(),
    }
    
    return results, y_pred, y_pred_proba


def main():
    """Main training pipeline."""
    print("\n" + "🔷"*35)
    print("BASELINE NLP MODELS - UNIFIED PIPELINE")
    print("🔷"*35)
    
    # Load data
    df = load_data_from_pipeline()
    
    # Split
    X_train, X_val, y_train, y_val = split_data(df)
    
    # Vectorize
    X_train_vec, X_val_vec, vectorizer = vectorize(X_train, X_val)
    
    # Train
    lr_model = train_logistic_regression(X_train_vec, y_train)
    svm_model = train_linear_svm(X_train_vec, y_train)
    
    # Evaluate
    lr_results, lr_pred, lr_proba = evaluate_model(lr_model, X_val_vec, y_val, "Logistic Regression")
    svm_results, svm_pred, svm_proba = evaluate_model(svm_model, X_val_vec, y_val, "Linear SVM")
    
    # Save models
    print(f"\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    lr_path = MODEL_DIR / "logistic_regression_baseline.joblib"
    svm_path = MODEL_DIR / "linear_svm_baseline.joblib"
    
    joblib.dump(lr_model, lr_path)
    joblib.dump(svm_model, svm_path)
    
    print(f"✓ Logistic Regression: {lr_path}")
    print(f"✓ Linear SVM: {svm_path}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(df),
        "models": [lr_results, svm_results],
        "vectorizer_features": len(vectorizer.vocabulary_),
        "data_split": {"train": len(X_train), "val": len(X_val)},
    }
    
    results_path = RESULTS_DIR / f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results: {results_path}")
    
    # Summary
    print(f"\n" + "✨"*35)
    print("TRAINING COMPLETE")
    print("✨"*35)
    print(f"\nBest model: {'Logistic Regression' if lr_results['roc_auc'] > svm_results['roc_auc'] else 'Linear SVM'}")
    print(f"  ROC-AUC: {max(lr_results['roc_auc'], svm_results['roc_auc']):.4f}")
    
    return results


if __name__ == "__main__":
    main()
