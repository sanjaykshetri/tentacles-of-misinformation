# Repository Structure — 4-Unit Framework

This document describes the current repository structure aligned with the 4-unit research framework for "The Tentacles of Misinformation" book project.

---

## 📖 Book (Primary Deliverable)

```
book/
├── _quarto.yml                    # Configuration: 6 chapters (prologue + 4 units + epilogue)
├── chapters/
│   ├── 00-prologue.qmd           # When a False Story Wins
│   ├── 01-measuring-vulnerability.qmd     # Unit 1: Behavioral Science
│   ├── 02-detecting-narratives.qmd        # Unit 2: NLP Pipeline
│   ├── 03-modeling-spread.qmd             # Unit 3: Epidemiology
│   ├── 04-fusion-scale.qmd                # Unit 4: Production & Ethics
│   └── 05-epilogue.qmd           # Frontier Challenges & Roadmap
├── index.qmd                      # Book homepage
├── references.bib                 # Bibliography
├── _book/                         # Generated HTML output (do not commit edits here)
├── CHAPTER_ROADMAP.md            # What each unit needs (visualization, code references)
└── render_output.txt              # Latest render log
```

**Live Site:** https://sanjaykshetri.github.io/tentacles-of-misinformation/

---

## 🧠 Unit 1: Behavioral Analysis
```
behavioral_analysis/
├── README.md                     # Overview of behavioral module
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb       # EDA, feature validation
│   ├── 02_baseline_models.ipynb            # Logistic regression, SVM
│   ├── 03_linguistic_features.ipynb        # NLP-behavioral interaction
│   ├── 03_regression_models.ipynb          # Statistical modeling
│   └── 04_tree_models_shap.ipynb           # SHAP feature importance
├── scripts/                      # Reusable Python modules
└── results/                      # Analysis outputs
```

**Maps to:** `book/chapters/01-measuring-vulnerability.qmd`

---

## 🤖 Unit 2: NLP Models
```
nlp_models/
├── README.md                     # Overview of NLP module
├── preprocessing/                # Text cleaning, tokenization
├── embeddings/                   # SBERT, transformer embeddings
├── classification/               # BERT, RoBERTa, DistilBERT classifiers
├── evaluation/                   # Metrics, cross-validation
└── notebooks/
    ├── 02_baseline_models.ipynb  # TF-IDF → Logistic Regression, SVM
    ├── 03_linguistic_features.ipynb
    └── 04_transformers.ipynb     # RoBERTa fine-tuning
```

**Maps to:** `book/chapters/02-detecting-narratives.qmd`

---

## 🔗 Unit 3 & 4: Fusion & Scale
```
fusion_models/
├── README.md                     # Overview of fusion module  
├── feature_engineering/          # Behavioral + NLP feature fusion
├── multimodal_models/            # Early/late fusion architectures
├── experiments/                  # SEIR simulations, hyperparameter tuning
└── notebooks/
    ├── 03_modeling_spread.ipynb  # SEIR calibration
    └── 04_ensemble.ipynb         # Fusion architecture

dashboards/
├── README.md                     # Overview of dashboards
├── streamlit/                    # Interactive Python dashboards
│   ├── app.py                    # Main dashboard entry point
│   ├── behavioral_explorer.py
│   ├── nlp_model_demo.py
│   └── risk_predictor.py
└── powerbi/                      # Business intelligence dashboards

src/
├── data_prep.py                  # Data loading and preprocessing
├── features.py                   # Feature extraction
├── train_baseline.py             # Train baseline models
├── train_linguistic_features.py  # Train NLP-behavioral models
└── train_transformers.py         # Train transformer models
```

**Maps to:**
- `book/chapters/03-modeling-spread.qmd` (Unit 3: SEIR simulations)
- `book/chapters/04-fusion-scale.qmd` (Unit 4: Dashboards & ethics)

---

## 📊 Data
```
data/
├── README.md                     # Data documentation
├── raw/
│   └── fakenewsnet/             # FakeNewsNet corpus
│       ├── gossipcop_fake.csv
│       ├── gossipcop_real.csv
│       ├── politifact_fake.csv
│       └── politifact_real.csv
└── processed/                    # Cleaned datasets, features
```

---

## 📈 Results & Models
```
results/
├── baseline_results_*.txt        # Model performance summaries
├── transformer_results_*.txt
├── model_comparison.png          # Ensemble performance chart
├── roc_*.png                     # ROC curves
├── cm_*.png                      # Confusion matrices
├── pr_*.png                      # Precision-recall curves
└── [other visualizations]

models/
├── behavioral_model.joblib
├── feature_scaler.joblib
├── hybrid_model.joblib
├── linear_svm.joblib
├── logistic_regression.joblib
└── tfidf_vectorizer.joblib
```

---

## 📚 Documentation
```
docs/
├── README.md                     # Overview of 4-unit framework
├── CAPSTONE_PROPOSAL.md          # Original project proposal
├── THESIS_INTEGRATION.md         # How thesis maps to 4 units
├── TECHNICAL_ROADMAP.md          # Development phases
├── figures/                      # Publication-ready plots (empty - generate from results/)
└── architecture_diagrams/        # System diagrams (empty - to be created)
```

---

## 🔧 Configuration & Deployment
```
environment/
├── conda.yml                     # Conda environment spec
├── requirements.txt              # Pip dependencies
└── README.md                     # Setup instructions

.github/
└── workflows/
    └── quarto-publish.yml        # Auto-deploy to GitHub Pages

LICENSE                           # MIT License
README.md                         # Main project overview
DEPLOYMENT_GUIDE.md               # How to render and deploy book
```

---

## 📦 Data Sources

| Dataset | Files | Records | Source |
|---------|-------|---------|--------|
| **FakeNewsNet** | 4 CSV | ~5K articles | `data/raw/fakenewsnet/` |
| **Behavioral Survey** | Python pickle | ~194 subjects | `behavioral_analysis/` |
| **LIAR** | (linkedfrom code) | ~13K statements | External (Hugging Face) |

---

## 🚀 Key Directories by Research Unit

### Unit 1 (Behavioral Science)
- **Source:** `behavioral_analysis/`
- **Output:** `book/chapters/01-measuring-vulnerability.qmd`
- **Visualization:** Descriptive stats, feature correlations, path diagrams

### Unit 2 (NLP Pipeline)
- **Source:** `nlp_models/`, `results/` (PNG files)
- **Output:** `book/chapters/02-detecting-narratives.qmd`
- **Visualization:** ROC curves, confusion matrices, model comparisons

### Unit 3 (Epidemiology)
- **Source:** `fusion_models/experiments/`
- **Output:** `book/chapters/03-modeling-spread.qmd`
- **Visualization:** SEIR dynamics, sensitivity heatmaps

### Unit 4 (Fusion & Scale)
- **Source:** `dashboards/`, `src/`, `fusion_models/`
- **Output:** `book/chapters/04-fusion-scale.qmd`
- **Visualization:** Architecture diagrams, dashboard screenshots, monitoring plots

---

## ✅ Repository Consistency Checklist

- [x] Book structure: 6 chapters (prologue + 4 units + epilogue)
- [x] Chapter files named sequentially: 00-prologue, 01-04 units, 05-epilogue
- [x] GitHub Pages deployed and live
- [x] Documentation aligned with 4-unit framework
- [x] README.md updated (no outdated chapter references)
- [x] CHAPTER_ROADMAP.md updated with visualization priorities
- [x] THESIS_INTEGRATION.md mapped to 4-unit model
- [x] Old chapter files (10-chapter structure) removed
- [ ] (Optional) Empty directories cleaned up (`nlp_pipelines/`, `research_book/`)
- [ ] (Optional) Architecture diagrams generated in `docs/architecture_diagrams/`

---

## 📝 File Cleanup Status

**Removed Files:**
- Old 10-chapter structure (01-introduction through 07-ethics-responsible-ds)
- Renamed 08-epilogue.qmd → 05-epilogue.qmd

**Empty/Unused Directories:**
- `nlp_pipelines/` — can be removed or reserved for future pipeline code
- `research_book/` — can be removed or used for paper manuscripts

**To Create:**
- `docs/architecture_diagrams/` — System design diagrams
- `docs/figures/` — Publication-ready visualization exports

---

## 🔗 Navigation

- **Book:** https://sanjaykshetri.github.io/tentacles-of-misinformation/
- **GitHub:** https://github.com/sanjaykshetri/tentacles-of-misinformation
- **Deployment Guide:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Thesis Integration:** [docs/THESIS_INTEGRATION.md](docs/THESIS_INTEGRATION.md)
- **Chapter Roadmap:** [book/CHAPTER_ROADMAP.md](book/CHAPTER_ROADMAP.md)

---

**Last updated:** April 1, 2026  
**Status:** Repository fully restructured for 4-unit framework ✅
