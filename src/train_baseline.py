"""
Sprint 2: Baseline NLP Models for FakeNewsNet
Trains TF-IDF + Logistic Regression and Linear SVM with class balancing.
Outputs metrics, confusion matrices, and model artifacts.
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# Paths - use absolute paths relative to script location
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "data" / "processed" / "articles.parquet"
MODEL_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and prepare data."""
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["label_num"] = (df["label"] == "fake").astype(int)
    print(f"Loaded {len(df)} articles")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Stratified train/validation split."""
    print(f"\nSplitting data (test_size={test_size})...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        df["title"],  # Using title as text feature
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
    """TF-IDF vectorization."""
    print("\nVectorizing with TF-IDF...")
    
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
    
    print(f"Vectorizer fitted on {len(vectorizer.vocabulary_)} features")
    print(f"Train matrix shape: {X_train_vec.shape}")
    print(f"Val matrix shape: {X_val_vec.shape}")
    
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    print(f"Vectorizer saved to {MODEL_DIR / 'tfidf_vectorizer.joblib'}")
    
    return X_train_vec, X_val_vec, vectorizer

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with balanced class weights."""
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    print("="*60)
    
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print("✅ Logistic Regression trained")
    
    return model

def train_linear_svm(X_train, y_train):
    """Train Linear SVM with balanced class weights."""
    print("\n" + "="*60)
    print("Training Linear SVM...")
    print("="*60)
    
    model = LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print("✅ Linear SVM trained")
    
    return model

def evaluate_model(model, X_val, y_val, model_name, vectorizer=None):
    """Comprehensive model evaluation."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_val)
    
    # For LogReg, get probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    else:
        # LinearSVC doesn't have predict_proba, use decision_function
        y_scores = model.decision_function(X_val)
        roc_auc = roc_auc_score(y_val, y_scores)
        y_pred_proba = 1 / (1 + np.exp(-y_scores))  # sigmoid
        print(f"\nROC-AUC Score (via decision_function): {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Real", "Fake"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Store results
    results = {
        "model_name": model_name,
        "y_true": y_val.values,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_val, y_pred, output_dict=True)
    }
    
    return results, y_pred, y_pred_proba

def plot_confusion_matrix(cm, model_name, output_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
        ax=ax, cbar_kws={'label': 'Count'}
    )
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_roc_curve(y_val, y_pred_proba, model_name, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}', fontweight='bold')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {output_path}")

def plot_precision_recall(y_val, y_pred_proba, model_name, output_path):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2, label=f'{model_name}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-recall curve saved to {output_path}")

def error_analysis(X_val, y_val, y_pred, model_name, top_n=5):
    """Analyze misclassified samples."""
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {model_name.upper()}")
    print(f"{'='*60}")
    
    errors_df = pd.DataFrame({
        'title': X_val.values,
        'true_label': y_val.values,
        'pred_label': y_pred
    })
    
    # False positives (real but predicted fake)
    false_pos = errors_df[(errors_df.true_label == 0) & (errors_df.pred_label == 1)]
    print(f"\nFalse Positives (Real → Fake): {len(false_pos)} samples")
    if len(false_pos) > 0:
        print("Examples:")
        for idx, row in false_pos.head(top_n).iterrows():
            print(f"  - {row['title'][:80]}")
    
    # False negatives (fake but predicted real)
    false_neg = errors_df[(errors_df.true_label == 1) & (errors_df.pred_label == 0)]
    print(f"\nFalse Negatives (Fake → Real): {len(false_neg)} samples")
    if len(false_neg) > 0:
        print("Examples:")
        for idx, row in false_neg.head(top_n).iterrows():
            print(f"  - {row['title'][:80]}")
    
    return errors_df

def save_results_summary(all_results):
    """Save results summary to text file."""
    summary_path = RESULTS_DIR / f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(summary_path, 'w') as f:
        f.write("FAKENEWSNET BASELINE MODEL RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"ROC-AUC: {result['roc_auc']:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{result['confusion_matrix']}\n")
            f.write(f"\nClassification Report:\n")
            f.write(str(result['classification_report']) + "\n")
            f.write("="*60 + "\n\n")
    
    print(f"\nResults summary saved to {summary_path}")

if __name__ == "__main__":
    print("\n" + "🐙"*10)
    print("SPRINT 2: BASELINE NLP MODELS")
    print("🐙"*10 + "\n")
    
    # Load data
    df = load_data()
    
    # Split
    X_train, X_val, y_train, y_val = split_data(df)
    
    # Vectorize
    X_train_vec, X_val_vec, vectorizer = vectorize(X_train, X_val)
    
    # Train models
    models = {
        "Logistic Regression": train_logistic_regression(X_train_vec, y_train),
        "Linear SVM": train_linear_svm(X_train_vec, y_train)
    }
    
    # Evaluate
    all_results = []
    for model_name, model in models.items():
        results, y_pred, y_pred_proba = evaluate_model(
            model, X_val_vec, y_val, model_name, vectorizer
        )
        all_results.append(results)
        
        # Save model
        model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Plots
        plot_confusion_matrix(
            results['confusion_matrix'],
            model_name,
            RESULTS_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
        )
        plot_roc_curve(
            y_val, y_pred_proba, model_name,
            RESULTS_DIR / f"roc_{model_name.lower().replace(' ', '_')}.png"
        )
        plot_precision_recall(
            y_val, y_pred_proba, model_name,
            RESULTS_DIR / f"pr_{model_name.lower().replace(' ', '_')}.png"
        )
        
        # Error analysis
        error_analysis(X_val, y_val, y_pred, model_name)
    
    # Save summary
    save_results_summary(all_results)
    
    print("\n" + "🔥"*10)
    print("SPRINT 2 COMPLETE")
    print("🔥"*10 + "\n")
