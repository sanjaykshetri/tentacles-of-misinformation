"""
Sprint 3: Train and Compare Three Models
- Behavioral features only
- TF-IDF baseline (loaded)
- Hybrid (TF-IDF + Behavioral)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, f1_score, accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# Paths
REPO_DIR = Path(__file__).parent.parent
DATA_PATH = REPO_DIR / "data" / "processed"
MODEL_DIR = REPO_DIR / "models"
RESULTS_DIR = REPO_DIR / "results"

DATA_PATH.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data_and_features():
    """Load articles and extracted features."""
    df = pd.read_parquet(DATA_PATH / "articles.parquet")
    feats = pd.read_parquet(DATA_PATH / "features.parquet")
    
    df["label_num"] = (df["label"] == "fake").astype(int)
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    
    print(f"Loaded {len(df)} articles with {feats.shape[1]} features")
    return df, feats.columns.tolist()

def split_data(df, feature_cols):
    """Create stratified train/val split."""
    X_train, X_val, y_train, y_val = train_test_split(
        df[["title"] + feature_cols],
        df["label_num"],
        test_size=0.2,
        stratify=df["label_num"],
        random_state=42
    )
    
    return X_train, X_val, y_train, y_val

def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model performance."""
    y_pred = model.predict(X_val)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_val)
        y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
    
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{model_name.upper()}")
    print(f"{'='*70}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return {
        "model_name": model_name,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": confusion_matrix(y_val, y_pred)
    }

def plot_comparison(results_dict, y_val):
    """Plot model comparison."""
    comparison_df = pd.DataFrame({
        'Model': [r['model_name'] for r in results_dict.values()],
        'Accuracy': [r['accuracy'] for r in results_dict.values()],
        'F1 Score': [r['f1'] for r in results_dict.values()],
        'ROC-AUC': [r['roc_auc'] for r in results_dict.values()]
    })
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(comparison_df))
    width = 0.25
    
    ax.bar(x - width, comparison_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x, comparison_df['F1 Score'], width, label='F1 Score', alpha=0.8)
    ax.bar(x + width, comparison_df['ROC-AUC'], width, label='ROC-AUC', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df

def plot_roc_curves(results_dict, y_val):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['purple', 'darkorange', 'darkgreen']
    for (key, result), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_val, result['y_pred_proba'])
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f"{result['model_name']} (AUC={result['roc_auc']:.3f})")
    
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison', fontweight='bold', fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("\n" + "🧠"*15)
    print("SPRINT 3: LINGUISTIC FEATURES COMPARISON")
    print("🧠"*15)
    
    # Load data
    df, feature_cols = load_data_and_features()
    X_train, X_val, y_train, y_val = split_data(df, feature_cols)
    
    # ============= MODEL 1: BEHAVIORAL ONLY =============
    print("\n[1/3] Training Behavioral-Only Model...")
    scaler = StandardScaler()
    X_train_behav = scaler.fit_transform(X_train[feature_cols])
    X_val_behav = scaler.transform(X_val[feature_cols])
    
    model_behav = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model_behav.fit(X_train_behav, y_train)
    result_behav = evaluate_model(model_behav, X_val_behav, y_val, "Behavioral Only")
    
    # ============= MODEL 2: TF-IDF (BASELINE) =============
    print("\n[2/3] Loading TF-IDF Baseline...")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model_tfidf = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    
    X_val_tfidf = vectorizer.transform(X_val["title"])
    result_tfidf = evaluate_model(model_tfidf, X_val_tfidf, y_val, "TF-IDF Baseline")
    
    # ============= MODEL 3: HYBRID =============
    print("\n[3/3] Training Hybrid Model...")
    X_train_tfidf = vectorizer.transform(X_train["title"])
    X_train_hybrid = hstack([X_train_tfidf, X_train_behav])
    X_val_hybrid = hstack([X_val_tfidf, X_val_behav])
    
    model_hybrid = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model_hybrid.fit(X_train_hybrid, y_train)
    result_hybrid = evaluate_model(model_hybrid, X_val_hybrid, y_val, "Hybrid (TF-IDF + Behavioral)")
    
    # ============= COMPARISON =============
    results = {
        "behav": result_behav,
        "tfidf": result_tfidf,
        "hybrid": result_hybrid
    }
    
    print("\n" + "="*70)
    print("SPRINT 3 SUMMARY")
    print("="*70)
    comparison_df = plot_comparison(results, y_val)
    print("\n" + comparison_df.to_string(index=False))
    
    # Show improvements
    baseline_auc = result_tfidf['roc_auc']
    print(f"\nImprovement over TF-IDF baseline ({baseline_auc:.4f}):")
    print(f"  • Behavioral: {((result_behav['roc_auc'] - baseline_auc) / baseline_auc * 100):+.2f}%")
    print(f"  • Hybrid: {((result_hybrid['roc_auc'] - baseline_auc) / baseline_auc * 100):+.2f}%")
    
    # Plot
    plot_roc_curves(results, y_val)
    
    # Save models
    joblib.dump(model_behav, MODEL_DIR / "behavioral_model.joblib")
    joblib.dump(model_hybrid, MODEL_DIR / "hybrid_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")
    
    print("\n✅ Models saved to models/")
    print("✅ Comparison plots saved to results/")
    
    print("\n" + "🔥"*15)
