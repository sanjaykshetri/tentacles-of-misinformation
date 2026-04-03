"""
SBERT Embedding-Based Misinformation Classifier

Uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings,
then trains classifiers (LogisticRegression, SVM, MLP) on top.
Optionally fine-tunes DistilBERT for sequence classification.

Usage:
    python src/train_sbert.py [--model sbert|distilbert] [--save]
"""

import sys
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load data via unified pipeline, falling back to synthetic demo data."""
    try:
        from data.pipeline import MisinformationPipeline, PipelineConfig
        config = PipelineConfig(verbose=False)
        processed_path = config.processed_data_dir / "articles_processed.parquet"
        if processed_path.exists():
            df = pd.read_parquet(processed_path)
            print(f"✓ Loaded {len(df)} articles from processed cache")
            return df, True
        pipeline = MisinformationPipeline(config)
        df = pipeline.run(save=True)
        print(f"✓ Pipeline ran, loaded {len(df)} articles")
        return df, True
    except Exception as e:
        print(f"Pipeline unavailable ({e}), generating synthetic data...")
        return _make_synthetic_data(), False


def _make_synthetic_data(n=2000, seed=42):
    """Generate synthetic headline data mirroring FakeNewsNet structure."""
    rng = np.random.default_rng(seed)
    real_templates = [
        "Scientists confirm {} in new peer-reviewed study",
        "Government announces {} funding for public health",
        "Report: {} sees steady improvement amid challenges",
        "Researchers find evidence of {} in long-term analysis",
        "{} wins award for contributions to community",
        "New study: {} linked to improved outcomes",
        "Officials clarify {} policy after public concerns",
        "Survey shows {} supports {} among majority of Americans",
    ]
    fake_templates = [
        "BREAKING: {} secretly {}, insiders reveal",
        "SHOCK: Scientists HIDE truth about {} from the public",
        "They don't want you to know this about {}",
        "MIRACLE: {} CURES everything overnight — Big Pharma panics",
        "EXPOSED: {} has been lying about {} for decades",
        "You won't believe what {} is doing behind closed doors",
        "URGENT: {} causes {} — share before they delete this",
        "{} CONFIRMS what we've known all along about {}",
    ]
    topics = [
        "climate change", "vaccines", "election results", "the economy",
        "healthcare", "immigration", "AI technology", "pharmaceutical drugs",
        "social media", "the moon landing", "water fluoridation", "5G towers",
    ]

    titles, labels = [], []
    for _ in range(n // 2):
        t = real_templates[rng.integers(len(real_templates))]
        t1 = topics[rng.integers(len(topics))]
        t2 = topics[rng.integers(len(topics))]
        try:
            titles.append(t.format(t1, t2))
        except IndexError:
            titles.append(t.format(t1))
        labels.append("real")

    for _ in range(n // 2):
        t = fake_templates[rng.integers(len(fake_templates))]
        t1 = topics[rng.integers(len(topics))]
        t2 = topics[rng.integers(len(topics))]
        try:
            titles.append(t.format(t1, t2))
        except IndexError:
            titles.append(t.format(t1))
        labels.append("fake")

    idx = rng.permutation(n)
    return pd.DataFrame({"title": [titles[i] for i in idx], "label": [labels[i] for i in idx]})


# ---------------------------------------------------------------------------
# SBERT embeddings
# ---------------------------------------------------------------------------

def get_sbert_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=64, cache_path=None):
    """Encode texts with SBERT; cache to disk if cache_path provided."""
    if cache_path and Path(cache_path).exists():
        print(f"✓ Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=DEVICE)
    print(f"Encoding {len(texts)} texts with {model_name} on {DEVICE}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if cache_path:
        np.save(cache_path, embeddings)
        print(f"✓ Saved embeddings to {cache_path}")
    return embeddings


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_classifiers(X_train, X_val, y_train, y_val):
    """Train LR, SVM, and MLP on top of SBERT embeddings."""
    classifiers = {
        "SBERT + LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "SBERT + SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
        ),
        "SBERT + MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }

    results = {}
    trained_models = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_val, preds),
            "f1": f1_score(y_val, preds, pos_label=1, average="binary"),
            "precision": precision_score(y_val, preds, pos_label=1, average="binary", zero_division=0),
            "recall": recall_score(y_val, preds, pos_label=1, average="binary", zero_division=0),
            "roc_auc": roc_auc_score(y_val, proba) if proba is not None else None,
        }
        results[name] = metrics
        trained_models[name] = clf

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "  ROC-AUC: N/A")

    return results, trained_models


# ---------------------------------------------------------------------------
# DistilBERT fine-tuning (optional)
# ---------------------------------------------------------------------------

def fine_tune_distilbert(train_df, val_df, label_col="label_num", text_col="title",
                          epochs=3, max_len=128, batch_size=16):
    """Fine-tune DistilBERT for sequence classification."""
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"Transformers/datasets not installed: {e}. Skipping fine-tuning.")
        return None

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning {model_name} on {device} for {epochs} epoch(s)...")

    def tokenize(examples):
        return tokenizer(examples[text_col], padding="max_length",
                         truncation=True, max_length=max_len)

    train_ds = Dataset.from_dict({"title": train_df[text_col].tolist(),
                                   "label": train_df[label_col].tolist()})
    val_ds = Dataset.from_dict({"title": val_df[text_col].tolist(),
                                  "label": val_df[label_col].tolist()})
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[text_col])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=[text_col])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary"),
            "roc_auc": roc_auc_score(labels, proba),
        }

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "distilbert"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        logging_steps=50,
        seed=42,
        no_cuda=(device == "cpu"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"DistilBERT evaluation: {eval_results}")

    # Save
    model.save_pretrained(MODEL_DIR / "distilbert_finetuned")
    tokenizer.save_pretrained(MODEL_DIR / "distilbert_finetuned")
    return {"DistilBERT (fine-tuned)": eval_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sbert", "distilbert", "both"], default="sbert")
    parser.add_argument("--save", action="store_true", help="Save trained models")
    args = parser.parse_args()

    print("=" * 70)
    print("TRANSFORMER-BASED MISINFORMATION CLASSIFIER")
    print("=" * 70)

    df, using_real = load_data()
    df["label_num"] = (df["label"] == "fake").astype(int)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label_num"], random_state=42
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    all_results = {}

    if args.model in ("sbert", "both"):
        cache_dir = MODEL_DIR / "embeddings"
        cache_dir.mkdir(exist_ok=True)

        X_train_emb = get_sbert_embeddings(
            train_df["title"].tolist(),
            cache_path=cache_dir / "train_sbert.npy"
        )
        X_val_emb = get_sbert_embeddings(
            val_df["title"].tolist(),
            cache_path=cache_dir / "val_sbert.npy"
        )

        y_train = train_df["label_num"].values
        y_val = val_df["label_num"].values

        sbert_results, trained_models = train_classifiers(X_train_emb, X_val_emb, y_train, y_val)
        all_results.update(sbert_results)

        if args.save:
            best_name = max(sbert_results, key=lambda k: sbert_results[k]["roc_auc"] or 0)
            best_model = trained_models[best_name]
            joblib.dump(best_model, MODEL_DIR / "sbert_best_classifier.joblib")
            np.save(MODEL_DIR / "embeddings" / "train_sbert.npy", X_train_emb)
            print(f"✓ Saved best SBERT classifier: {best_name}")

    if args.model in ("distilbert", "both"):
        distilbert_results = fine_tune_distilbert(train_df, val_df)
        if distilbert_results:
            all_results.update(distilbert_results)

    # Save results summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"transformer_results_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({"timestamp": ts, "device": DEVICE, "models": all_results}, f, indent=2)
    print(f"\n✓ Results saved to {summary_path}")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    for name, m in all_results.items():
        auc = m.get("roc_auc") or m.get("eval_roc_auc", "N/A")
        acc = m.get("accuracy") or m.get("eval_accuracy", "N/A")
        print(f"  {name:<40} Acc={acc:.4f}  AUC={auc:.4f}" if isinstance(auc, float) else f"  {name}")


if __name__ == "__main__":
    main()
