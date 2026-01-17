"""
Sprint 4: Transformer Fine-tuning on RoBERTa
Simplified version for faster execution
"""

import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from datetime import datetime

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
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

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_data_and_split():
    """Load articles and create train/val split."""
    df = pd.read_parquet(DATA_PATH / "articles.parquet")
    df["label_num"] = (df["label"] == "fake").astype(int)
    
    # 80/20 stratified split (same as before for consistency)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label_num"],
        random_state=42
    )
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Train label distribution: {train_df['label_num'].value_counts().to_dict()}")
    
    return train_df, val_df

def create_huggingface_dataset(df, tokenizer, max_length=128):
    """Convert dataframe to HuggingFace Dataset."""
    def tokenize_function(examples):
        return tokenizer(
            examples["title"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_dict({
        "title": df["title"].tolist(),
        "label": df["label_num"].tolist()
    })
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["title"])
    return dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }

def setup_lora_model(model_name="roberta-base"):
    """Setup RoBERTa with LoRA for efficient fine-tuning."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        device_map="auto" if DEVICE == "cuda" else None
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def train_transformer(train_dataset, val_dataset, tokenizer):
    """Fine-tune RoBERTa with LoRA."""
    print("\n" + "="*70)
    print("FINE-TUNING ROBERTA WITH LORA")
    print("="*70)
    
    model = setup_lora_model("roberta-base")
    
    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "transformer_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(RESULTS_DIR / "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=2e-4,
        seed=42,
        fp16=DEVICE == "cuda",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    
    return model, trainer

def evaluate_transformer(model, val_dataset, val_df, tokenizer):
    """Evaluate transformer on validation set."""
    print("\n" + "="*70)
    print("TRANSFORMER EVALUATION")
    print("="*70)
    
    # Get predictions
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(val_dataset, batch_size=32):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    y_pred = np.argmax(logits, axis=1)
    y_pred_proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
    
    # Metrics
    accuracy = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    roc_auc = roc_auc_score(labels, y_pred_proba)
    cm = confusion_matrix(labels, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "cm": cm,
        "labels": labels
    }

def extract_attention_weights(model, val_dataset, tokenizer, num_examples=5):
    """Extract attention weights for interpretability."""
    print("\n" + "="*70)
    print("ATTENTION WEIGHT ANALYSIS")
    print("="*70)
    
    model.eval()
    attention_data = []
    
    with torch.no_grad():
        for i, batch in enumerate(torch.utils.data.DataLoader(val_dataset, batch_size=1)):
            if i >= num_examples:
                break
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Average attention across heads and layers
            attentions = outputs.attentions
            avg_attention = torch.stack([att.squeeze(0).mean(0) for att in attentions]).mean(0)
            
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            attention_data.append({
                "tokens": tokens,
                "attention": avg_attention.cpu().numpy(),
                "label": batch["labels"][0].item()
            })
    
    print(f"Extracted attention for {len(attention_data)} examples")
    return attention_data

def plot_transformer_comparison(transformer_results, baseline_results):
    """Compare transformer vs classical models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics comparison
    models = ["Behavioral\nOnly", "TF-IDF\nBaseline", "Hybrid\n(TF-IDF+Behav)", "RoBERTa\nTransformer"]
    roc_aucs = [
        baseline_results["behav_auc"],
        baseline_results["tfidf_auc"],
        baseline_results["hybrid_auc"],
        transformer_results["roc_auc"]
    ]
    
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"]
    bars = ax1.bar(models, roc_aucs, color=colors, alpha=0.8)
    ax1.set_ylabel("ROC-AUC Score", fontsize=12)
    ax1.set_title("Model Performance Comparison", fontweight="bold", fontsize=13)
    ax1.set_ylim([0.5, 0.95])
    ax1.axhline(y=0.859, color="gray", linestyle="--", alpha=0.5, label="Classical Ceiling")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars, roc_aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # ROC curves
    fpr, tpr, _ = roc_curve(transformer_results["labels"], transformer_results["y_pred_proba"])
    ax2.plot(fpr, tpr, color="#d62728", lw=3, 
            label=f"RoBERTa (AUC={transformer_results['roc_auc']:.4f})")
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("RoBERTa ROC Curve", fontweight="bold", fontsize=13)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "transformer_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved comparison to {RESULTS_DIR / 'transformer_comparison.png'}")

def save_results(transformer_results, val_df):
    """Save results summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = f"""
SPRINT 4: TRANSFORMER FINE-TUNING RESULTS
{'='*70}

MODEL: RoBERTa with LoRA Fine-tuning
TRAINING: 3 epochs, batch_size=32, lr=2e-4
REGULARIZATION: LoRA (r=8, alpha=32)
DEVICE: {DEVICE}

PERFORMANCE METRICS:
  • Accuracy:  {transformer_results['accuracy']:.4f}
  • F1 Score:  {transformer_results['f1']:.4f}
  • ROC-AUC:   {transformer_results['roc_auc']:.4f}

CONFUSION MATRIX:
{transformer_results['cm']}

COMPARISON TO CLASSICAL MODELS:
  • Behavioral-Only:        0.6054 AUC
  • TF-IDF Baseline:        0.8590 AUC (Sprint 2)
  • Hybrid (TF-IDF+Behav):  0.8621 AUC (Sprint 3)
  • RoBERTa Transformer:    {transformer_results['roc_auc']:.4f} AUC 🎯

IMPROVEMENT:
  • vs TF-IDF baseline:     {((transformer_results['roc_auc'] - 0.859) / 0.859 * 100):+.2f}%
  • vs Hybrid model:        {((transformer_results['roc_auc'] - 0.8621) / 0.8621 * 100):+.2f}%

NEXT STEPS:
1. Create Quarto chapter 05-transformers.qmd with methodology and results
2. Implement SHAP explainability for feature importance
3. Analyze failure cases: where does RoBERTa fail that classical models succeed?
4. Prepare production deployment pipeline
"""
    
    with open(RESULTS_DIR / f"transformer_results_{timestamp}.txt", "w") as f:
        f.write(summary)
    
    print(summary)

if __name__ == "__main__":
    print("\n" + "🚀"*20)
    print("SPRINT 4: TRANSFORMER FINE-TUNING")
    print("🚀"*20)
    
    # Load data
    train_df, val_df = load_data_and_split()
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Create datasets
    train_dataset = create_huggingface_dataset(train_df, tokenizer)
    val_dataset = create_huggingface_dataset(val_df, tokenizer)
    
    # Fine-tune
    model, trainer = train_transformer(train_dataset, val_dataset, tokenizer)
    
    # Evaluate
    transformer_results = evaluate_transformer(model, val_dataset, val_df, tokenizer)
    
    # Extract attention (optional - for interpretability)
    attention_data = extract_attention_weights(model, val_dataset, tokenizer, num_examples=3)
    
    # Load baseline results for comparison
    baseline_results = {
        "behav_auc": 0.6054,
        "tfidf_auc": 0.8590,
        "hybrid_auc": 0.8621
    }
    
    # Plot comparison
    plot_transformer_comparison(transformer_results, baseline_results)
    
    # Save results
    save_results(transformer_results, val_df)
    
    # Save model
    model.save_pretrained(MODEL_DIR / "roberta_fine_tuned")
    tokenizer.save_pretrained(MODEL_DIR / "roberta_tokenizer")
    
    print("\n✅ Models saved to models/roberta_fine_tuned")
    print("✅ Results saved to results/")
    
    print("\n" + "🔥"*20)
