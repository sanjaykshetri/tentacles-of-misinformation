#!/usr/bin/env python3
"""
Master runner: bootstrap trained models + execute all Phase 2-3 notebooks.
Uses synthetic data everywhere (no FakeNewsNet CSVs required).
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Synthetic data + baseline model training
# ─────────────────────────────────────────────────────────────────────────────
def build_baseline_models():
    print("\n" + "="*60)
    print("STEP 1 — Training baseline NLP models (synthetic data)")
    print("="*60)

    rng = np.random.default_rng(42)
    n   = 5000

    real_templates = [
        "Scientists confirm {} in peer-reviewed study published in {}",
        "Government officials announce new {} policy affecting {} citizens",
        "Research shows {} linked to {} according to medical journal",
        "University study finds {} reduces risk of {} by {}%",
        "Health experts recommend {} to prevent {} spread",
    ]
    fake_templates = [
        "BREAKING: {} secretly {} — what they don't want you to know",
        "EXPOSED: {} HIDES {} from public — share before deleted",
        "SHOCKING: {} causes {} — mainstream media silent",
        "The truth about {} and {} that THEY are hiding from you",
        "MUST SEE: {} proves {} is all a hoax — spread the word",
    ]
    topics   = ["climate change","vaccines","5G towers","elections","cancer cures",
                "government spending","immigration","crime statistics","drug trials","AI technology"]
    journals = ["Nature","Lancet","NEJM","Science","JAMA","Cell","PNAS","BMJ"]
    nums     = [str(x) for x in range(10, 90, 5)]

    titles, labels = [], []
    for _ in range(n // 2):
        tmpl = real_templates[rng.integers(len(real_templates))]
        try:
            titles.append(tmpl.format(
                topics[rng.integers(len(topics))],
                journals[rng.integers(len(journals))],
                nums[rng.integers(len(nums))],
            ))
        except Exception:
            titles.append(tmpl.format(topics[rng.integers(len(topics))], "experts"))
        labels.append(0)

    for _ in range(n // 2):
        tmpl = fake_templates[rng.integers(len(fake_templates))]
        try:
            titles.append(tmpl.format(
                topics[rng.integers(len(topics))],
                topics[rng.integers(len(topics))],
            ))
        except Exception:
            titles.append(tmpl.format(topics[rng.integers(len(topics))]))
        labels.append(1)

    df = pd.DataFrame({"title": titles, "label_num": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train, X_val, y_train, y_val = train_test_split(
        df["title"], df["label_num"], test_size=0.2,
        stratify=df["label_num"], random_state=42
    )

    # TF-IDF vectoriser
    tfidf = TfidfVectorizer(
        stop_words="english", max_features=10000,
        ngram_range=(1, 2), sublinear_tf=True, min_df=2
    )
    X_tr_vec = tfidf.fit_transform(X_train)
    X_va_vec = tfidf.transform(X_val)

    # Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="saga", n_jobs=-1)
    lr.fit(X_tr_vec, y_train)
    lr_proba = lr.predict_proba(X_va_vec)[:, 1]
    lr_auc   = roc_auc_score(y_val, lr_proba)
    lr_acc   = accuracy_score(y_val, lr.predict(X_va_vec))
    print(f"  LR  — AUC: {lr_auc:.4f}  Acc: {lr_acc:.4f}")

    # SVM (calibrated for predict_proba support)
    svm_raw = LinearSVC(C=1.0, max_iter=2000)
    svm = CalibratedClassifierCV(svm_raw, cv=5)
    svm.fit(X_tr_vec, y_train)
    svm_proba = svm.predict_proba(X_va_vec)[:, 1]
    svm_auc   = roc_auc_score(y_val, svm_proba)
    svm_acc   = accuracy_score(y_val, svm.predict(X_va_vec))
    print(f"  SVM — AUC: {svm_auc:.4f}  Acc: {svm_acc:.4f}")

    # Save models
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(lr,    MODEL_DIR / "logistic_regression_baseline.joblib")
    joblib.dump(svm,   MODEL_DIR / "svm_baseline.joblib")
    print(f"\n  ✓ Models saved → {MODEL_DIR}")

    # Save baseline results JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "data_source": "synthetic",
        "n_train": len(X_train),
        "n_val":   len(X_val),
        "models": {
            "logistic_regression": {"auc": round(lr_auc, 4), "accuracy": round(lr_acc, 4)},
            "svm":                 {"auc": round(svm_auc, 4), "accuracy": round(svm_acc, 4)},
        }
    }
    with open(RESULTS_DIR / "baseline_summary_run.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  ✓ Results saved")
    return tfidf


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run notebooks with nbconvert
# ─────────────────────────────────────────────────────────────────────────────
NOTEBOOKS = [
    # Phase 2
    "nlp_models/notebooks/02_transformer_classifiers.ipynb",
    "nlp_models/notebooks/03_error_analysis.ipynb",
    "nlp_models/notebooks/04_model_comparison.ipynb",
    # Phase 3
    "fusion_models/notebooks/01_feature_alignment.ipynb",
    "fusion_models/notebooks/02_fusion_architectures.ipynb",
    "fusion_models/notebooks/03_ablation_studies.ipynb",
    "fusion_models/notebooks/04_interpretability.ipynb",
    "fusion_models/notebooks/05_validation_and_generalization.ipynb",
]


def run_notebook(nb_path: str) -> bool:
    path = PROJECT_ROOT / nb_path
    if not path.exists():
        print(f"  SKIP (not found): {nb_path}")
        return False
    print(f"\n  ▶ {nb_path}")
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            "--ExecutePreprocessor.kernel_name=python3",
            str(path),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print(f"     ✓ Done")
        return True
    else:
        print(f"     ✗ FAILED")
        # Show last 15 lines of stderr
        lines = result.stderr.strip().split("\n")
        for line in lines[-15:]:
            print(f"       {line}")
        return False


def run_all_notebooks():
    print("\n" + "="*60)
    print("STEP 2 — Running all notebooks")
    print("="*60)
    passed, failed = [], []
    for nb in NOTEBOOKS:
        ok = run_notebook(nb)
        (passed if ok else failed).append(nb)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Passed : {len(passed)}")
    print(f"  Failed : {len(failed)}")
    if failed:
        print("\n  Failed notebooks:")
        for f in failed:
            print(f"    • {f}")
    return len(failed) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = datetime.now()
    print(f"\nMaster runner started at {t0.strftime('%H:%M:%S')}")

    build_baseline_models()
    success = run_all_notebooks()

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'✓ All done' if success else '✗ Some notebooks failed'} — {elapsed:.0f}s elapsed")
    sys.exit(0 if success else 1)
