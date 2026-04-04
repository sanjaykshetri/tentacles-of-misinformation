# 🐙 Tentacles of Misinformation

### Detecting, Modeling, and Predicting Misinformation Using Behavioral Signals and NLP

> **Who falls for misinformation, why they do, and how we can detect and predict it at scale.**

An end-to-end data science system that combines **behavioral research, natural language processing, and epidemic modeling** to understand and combat misinformation.

---

## 🔗 Live Deployments

| Resource | Link |
|---|---|
| 📖 Research Book | [sanjaykshetri.github.io/tentacles-of-misinformation](https://sanjaykshetri.github.io/tentacles-of-misinformation/) |
| 🚀 Interactive App | [huggingface.co/spaces/sanjaykshetri/tentacles](https://huggingface.co/spaces/sanjaykshetri/tentacles) |

---

## 🧩 System Architecture

```
Human Behavior → Feature Engineering → NLP Models → Hybrid Fusion → Prediction → Spread Modeling
```

| Layer | Description |
|---|---|
| 🧠 Behavioral | Cognitive + psychological predictors of susceptibility |
| 🤖 NLP | Text-based misinformation detection (TF-IDF → Transformers) |
| 🔥 Fusion | Combined linguistic + behavioral features |
| 🌍 Spread | SEIR epidemic model calibrated to cascade data |

---

## 📊 Results

All numbers are from the held-out validation set (N=4,345, 80/20 stratified split on FakeNewsNet, 21,724 articles).

| Model | Accuracy | F1 | ROC-AUC | Speed |
|---|---|---|---|---|
| Behavioral-only | 55.5% | 0.399 | 0.607 | <1ms |
| **TF-IDF + LR** (baseline) | **81.2%** | **0.644** | **0.859** | 0.1ms |
| TF-IDF + SVM | 79.4% | 0.615 | 0.841 | 0.1ms |
| Hybrid (TF-IDF + Behavioral) | 81.3% | 0.645 | 0.863 | 0.2ms |
| **RoBERTa-base + LoRA** | **82.3%** | **0.659** | **0.870** | 8ms |

**Statistical validation:** McNemar test (Fusion vs. NLP-only) — χ²=43.37, p=4.5×10⁻¹¹

---

## 🧠 1. Modeling Human Vulnerability

Based on my Master's thesis and an IRB-approved behavioral study (N=194):

- Cognitive Reflection Test (CRT) predicts information verification behavior (β=0.149, p=.031)
- Psycholinguistic features (certainty markers, hedging, subjectivity) capture how manipulation targets cognition
- Susceptibility is **not random — it is measurable and predictable**

These behavioral priors feed directly into the feature engineering pipeline.

---

## 🤖 2. Misinformation Detection Engine

### What I Built

- **Sprint 2:** TF-IDF vectorizer (6,422 tokens, bigrams) + Logistic Regression and Linear SVM baselines
- **Sprint 3:** Psycholinguistic feature extraction (13 features: sentiment, readability, certainty/hedging markers) + hybrid fusion
- **Sprint 4:** RoBERTa-base fine-tuned with LoRA (r=8, α=32) — 0.71% of parameters trained

### Key Takeaway

The TF-IDF baseline reaches 98.6% of the transformer's AUC at 0.1ms vs. 8ms per article. The cost/performance tradeoff strongly favors baselines at production scale.

---

## 🔥 3. Hybrid Fusion (Core Contribution)

> **Text alone is not enough. Humans are part of the system.**

Combining linguistic signals with behavioral features (sentiment, readability, certainty markers) yields a statistically significant improvement over NLP-only (McNemar p=4.5×10⁻¹¹), even though the AUC gain is modest (+0.4pp). The behavioral features add robustness — they regularize the model against sensationalist real news that confuses text-only classifiers.

---

## 🌍 4. Misinformation as an Epidemic

Using a SEIR epidemiological model calibrated to FakeNewsNet cascade data:

| Parameter | Value | Interpretation |
|---|---|---|
| β (transmission rate) | 0.0153 | Low but nonzero per-contact spread |
| γ (recovery rate) | 0.0870 | ~11-day average exposure period |
| R₀ | 0.176 | Does not self-sustain without amplification |

**Intervention simulation:** Reducing β by 20% and increasing γ by 30% (friction + counter-narrative) reduces peak "infected" population by ~23% versus no intervention. Timing of intervention matters more than magnitude.

---

## ⚡ 5. Pipeline & Reproducibility

```
Raw CSVs (FakeNewsNet, 4 files)
    ↓ data/pipeline/loader.py        → 23,196 articles
    ↓ data/pipeline/cleaner.py       → 21,724 articles (1,472 removed, 6.3%)
    ↓ data/pipeline/transformers.py  → 13 psycholinguistic features
    ↓ data/pipeline/orchestrator.py  → saved to data/processed/*.parquet
    ↓ src/train_baseline_tracked.py  → model artifacts + MLflow logs
```

- Fixed random seeds throughout
- Versioned dependencies (`environment/requirements.txt`)
- Full experiment tracking (MLflow with local JSON fallback)
- 46+ unit tests across linked repositories

---

## ⚖️ 6. Responsible AI

- **Dataset bias:** GossipCop (entertainment) vs. PolitiFact (political) have different label distributions; cross-domain transfer estimated at 50–70% on health/science domains
- **False positive analysis:** Sensational real news is the primary source of misclassification
- **Explainability:** Top TF-IDF coefficients surfaced per prediction in the live app
- **Human-in-the-loop:** High-stakes flagging decisions are not fully automated

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13, R (behavioral analysis), Quarto (docs) |
| ML | scikit-learn, PyTorch, transformers (Hugging Face), peft |
| Feature engineering | VADER, TextBlob, textstat |
| Experiment tracking | MLflow (+ custom `ExperimentTracker`) |
| Deployment | Streamlit on Hugging Face Spaces, Quarto → GitHub Pages |
| Data | FakeNewsNet: 21,724 articles · IRB survey: N=194 |

---

## 🚀 Quick Start

**Prerequisites**
```bash
git clone https://github.com/sanjaykshetri/tentacles-of-misinformation.git
cd tentacles-of-misinformation
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r environment/requirements.txt
```

**Train baselines**
```powershell
# Windows
.\tasks.ps1 -Task train-tracked
```
```bash
# macOS/Linux
make train-tracked
```

What happens: downloads and processes 23K articles (~45 seconds), trains LR and SVM with experiment tracking (~2 minutes), saves models and metrics to `models/` and `results/`.

**Run the Streamlit app locally**
```bash
streamlit run dashboards/streamlit/research_hub.py
```
Open http://localhost:8501 — live prediction, SEIR simulator, model explorer, data browser.

**Build the book locally**
```bash
cd book
quarto render
```
Output: `book/_book/index.html`

---

## 🏗️ Repository Structure

```
tentacles-of-misinformation/
├── 📘 book/                          ← Quarto research book (6 chapters, 14,654 words)
│   ├── chapters/                     ← 00-prologue through 06-epilogue
│   ├── index.qmd                     ← Introduction
│   └── _book/                        ← Rendered HTML (GitHub Pages)
│
├── 🔧 data/pipeline/                 ← Unified data pipeline (4 modules)
│   ├── loader.py
│   ├── cleaner.py
│   ├── transformers.py
│   └── orchestrator.py
│
├── 🤖 src/                           ← Training scripts + utilities
│   ├── train_baseline.py             ← LR + SVM (Sprint 2)
│   ├── train_baseline_tracked.py     ← With MLflow tracking
│   ├── train_linguistic_features.py  ← Behavioral + Hybrid (Sprint 3)
│   ├── train_transformers.py         ← RoBERTa + LoRA (Sprint 4)
│   ├── features.py                   ← Psycholinguistic feature extraction
│   └── experiment_tracker.py         ← MLflow wrapper
│
├── 🎯 models/                        ← Trained artifacts
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression_baseline.joblib
│   └── linear_svm_baseline.joblib
│
├── 📊 dashboards/streamlit/          ← Research Hub app
│   └── research_hub.py
│
├── 📈 results/                       ← Metrics, curves, experiment logs
├── 📓 capstone_notebook.ipynb        ← Full pipeline walkthrough (38 cells)
├── environment/requirements.txt      ← Dependencies
├── Makefile                          ← Task automation (macOS/Linux)
├── tasks.ps1                         ← Task automation (Windows)
└── INTEGRATION_GUIDE.md              ← Pipeline documentation
```

---

## 🔗 Linked Repositories

| Repository | Purpose |
|---|---|
| [Misinformation-study-Masters-Thesis](https://github.com/sanjaykshetri/Misinformation-study-Masters-Thesis) | Behavioral study + survey methodology (N=194) |
| [Misinformation-Detection-ML-Model2](https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2) | NLP classifiers (TF-IDF → Transformers) |
| [misinformation-epidemic-model](https://github.com/sanjaykshetri/misinformation-epidemic-model) | SEIR spread modeling + cascade calibration |
| [misinformation-at-scale](https://github.com/sanjaykshetri/misinformation-at-scale) | Production pipelines + dashboards |

---

## 👤 About

**Sanjay Kumar Chhetri** — [sanjaykshetri@gmail.com](mailto:sanjaykshetri@gmail.com)

Background in psychology and data science. Interested in applying behavioral science to machine learning and building systems that are accurate, interpretable, and responsible.

---

## 📜 License

**Code:** MIT License  
**Data:** FakeNewsNet (per their license) · Behavioral survey (IRB-approved, anonymized)

---

*Built 2025–2026 · Actively maintained*
