"""
Baseline NLP Models with Experiment Tracking

Uses unified pipeline and MLflow for reproducible experiment management.

Usage:
    python src/train_baseline_tracked.py
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
    accuracy_score, f1_score, precision_score, recall_score
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline import MisinformationPipeline, PipelineConfig
from src.experiment_tracker import get_tracker

# Paths
SCRIPT_DIR = PROJECT_ROOT
MODEL_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_data_from_pipeline():
    """Load data using unified pipeline."""
    config = PipelineConfig(verbose=False)
    pipeline = MisinformationPipeline(config)
    
    processed_path = config.processed_data_dir / "articles_processed.parquet"
    
    if processed_path.exists():
        df = pd.read_parquet(processed_path)
    else:
        df = pipeline.run(save=True)
    
    df["label_num"] = (df["label"] == "fake").astype(int)
    
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Stratified train/validation split."""
    X_train, X_val, y_train, y_val = train_test_split(
        df["title"],
        df["label_num"],
        test_size=test_size,
        stratify=df["label_num"],
        random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val


def vectorize(X_train, X_val):
    """TF-IDF vectorization."""
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
    
    # Save vectorizer
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    
    return X_train_vec, X_val_vec, vectorizer


def train_and_evaluate(model_class, X_train, X_val, y_train, y_val, model_name, tracker):
    """Train model and evaluate with experiment tracking."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    
    with tracker.run(run_name=model_name, description=f"Baseline {model_name}"):
        
        # Log parameters
        params = {
            f"{model_name.lower().replace(' ', '_')}_class": model_class.__name__,
            "dataset_size": X_train.shape[0] + X_val.shape[0],
            "train_size": X_train.shape[0],
            "val_size": X_val.shape[0],
            "vectorizer": "TfidfVectorizer",
            "max_features": 10000,
            "ngram_range": "(1, 2)",
        }
        
        if model_name == "Logistic Regression":
            params.update({
                "max_iter": 2000,
                "solver": "lbfgs",
                "class_weight": "balanced",
            })
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
                n_jobs=-1,
            )
        else:  # Linear SVM
            params.update({
                "max_iter": 5000,
                "class_weight": "balanced",
            })
            model = LinearSVC(
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            )
        
        tracker.log_params(params)
        
        # Train
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        print(f"✓ Training complete")
        
        # Predict
        y_pred = model.predict(X_val)
        y_pred_proba = model.decision_function(X_val)
        y_pred_proba_norm = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min() + 1e-8)
        
        # Compute metrics
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_score": float(f1_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred)),
            "recall": float(recall_score(y_val, y_pred)),
            "roc_auc": float(roc_auc_score(y_val, y_pred_proba_norm)),
        }
        
        tracker.log_metrics(metrics)
        
        # Print results
        print(f"\nMetrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  • {metric_name}: {metric_value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:5d} | FP: {cm[0,1]:5d}")
        print(f"  FN: {cm[1,0]:5d} | TP: {cm[1,1]:5d}")
        
        # Save model
        model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}_baseline.joblib"
        joblib.dump(model, model_path)
        tracker.log_artifact(str(model_path))
        print(f"✓ Model saved: {model_path}")
        
        return model, metrics


def main():
    """Main training pipeline with experiment tracking."""
    
    print("\n" + "🔷"*35)
    print("BASELINE NLP MODELS WITH EXPERIMENT TRACKING")
    print("🔷"*35)
    
    # Initialize tracker
    tracker = get_tracker(use_mlflow=True, experiment_name="misinformation_detection")
    
    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    df = load_data_from_pipeline()
    print(f"✓ Loaded {len(df)} articles")
    
    # Split
    X_train, X_val, y_train, y_val = split_data(df)
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Vectorize
    print(f"\n{'='*70}")
    print("VECTORIZING")
    print(f"{'='*70}")
    X_train_vec, X_val_vec, vectorizer = vectorize(X_train, X_val)
    print(f"✓ Features: {X_train_vec.shape[1]}")
    
    # Train models
    results = {}
    
    lr_model, lr_metrics = train_and_evaluate(
        LogisticRegression,
        X_train_vec, X_val_vec, y_train, y_val,
        "Logistic Regression",
        tracker
    )
    results["logistic_regression"] = lr_metrics
    
    svm_model, svm_metrics = train_and_evaluate(
        LinearSVC,
        X_train_vec, X_val_vec, y_train, y_val,
        "Linear SVM",
        tracker
    )
    results["linear_svm"] = svm_metrics
    
    # Summary
    print(f"\n{'✨'*35}")
    print("TRAINING COMPLETE")
    print(f"{'✨'*35}")
    
    best_model = "logistic_regression" if lr_metrics["roc_auc"] > svm_metrics["roc_auc"] else "linear_svm"
    best_auc = max(lr_metrics["roc_auc"], svm_metrics["roc_auc"])
    print(f"\nBest Model: {best_model.replace('_', ' ').title()}")
    print(f"  ROC-AUC: {best_auc:.4f}")
    
    # Save results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_articles": len(df),
            "train_size": len(X_train),
            "val_size": len(X_val),
        },
        "models": results,
        "best_model": best_model,
        "best_auc": best_auc,
    }
    
    summary_path = RESULTS_DIR / f"baseline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved: {summary_path}")
    
    return summary


if __name__ == "__main__":
    main()
