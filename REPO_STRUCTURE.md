# Repository Structure — Hub & Spoke Model

This **monograph** (hub) integrates four specialized, independent repositories (spokes) into a cohesive research narrative published as an interactive Quarto book.

The hub coordinates, links, and showcases work from four dedicated external repos—each repository is self-contained and can be used independently.

---

## 🌐 Hub Architecture

```
        ┌─────────────────────────────────────────┐
        │  tentacles-of-misinformation (Hub)     │
        │    Published Quarto Research Book       │
        │  https://sanjaykshetri.github.io/...   │
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ↓              ↓              ↓              ↓
    ┌────────┐    ┌────────┐    ┌────────┐    ┌──────────┐
    │Unit 1  │    │Unit 2  │    │Unit 3  │    │Unit 4    │
    │(Spoke) │    │(Spoke) │    │(Spoke) │    │(Spoke)   │
    └────────┘    └────────┘    └────────┘    └──────────┘
        │              │              │              │
        ↓              ↓              ↓              ↓
   Misinform-    Detection ML    Epidemic      At Scale
   ation Study    Models          Model
   Masters Thesis
```

---

## 🔗 External Source Repositories (Spokes)

> **Key Principle:** Each spoke repository is a complete, independent project. The hub book coordinates and showcases their outputs.

### **Spoke 1: Behavioral-Cognitive Foundation**
**Repository:** [Misinformation-study-Masters-Thesis](https://github.com/sanjaykshetri/Misinformation-study-Masters-Thesis)

- **Language:** R (statistical analysis)
- **Data:** N=194 behavioral survey (Qualtrics), IRB-approved
- **Methods:** Mediation analysis, SEM (lavaan), measurement theory
- **Key Finding:** CRT predicts verification behavior (β=0.149, p=.031)
- **Deliverables:** Masters thesis PDF, analysis scripts, data dictionary, raw datasets
- **Maps to:** `book/chapters/01-measuring-vulnerability.qmd`
- **How to use:** Run `data_analysis1.R` for full reproducible pipeline

### **Spoke 2: NLP Baseline Models**
**Repository:** [Misinformation-Detection-ML-Model2](https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2)

- **Language:** Python (Jupyter)
- **Dataset:** FakeNewsNet (23,196 real fact-checked articles)
- **Methods:** TF-IDF + 3 classifiers (Logistic Regression, Random Forest, Gradient Boosting)
- **Results:** 83.62% accuracy (LR best), 87.83% ROC-AUC, rigorous statistical validation
- **Key Notebook:** `misinformation_analysis.ipynb` (complete pipeline)
- **Maps to:** `book/chapters/02-detecting-narratives.qmd` (Unit 2: Baseline)
- **How to use:** Open notebook, run cell-by-cell (includes EDA, training, evaluation)

### **Spoke 3: Epidemiological SEIR Modeling**
**Repository:** [misinformation-epidemic-model](https://github.com/sanjaykshetri/misinformation-epidemic-model)

- **Language:** Python (ODE-based simulation)
- **Model:** SEIR compartmental model (Susceptible→Exposed→Infected→Recovered)
- **Calibration:** FakeNewsNet cascade data (β=0.0153, σ=0.3193, γ=0.10)
- **Quality:** 46 unit tests, mypy strict type checking, CI/CD pipeline
- **Key Notebooks:** 
  - `notebooks/quick_start_academic.ipynb` (5-minute overview)
  - `notebooks/baseline_vs_interventions.ipynb` (full analysis)
- **Maps to:** `book/chapters/03-modeling-spread.qmd`
- **How to use:** Run `quick_start_academic.ipynb` for immediate results

### **Spoke 4: Production-Scale Deep Learning**
**Repository:** [misinformation-at-scale](https://github.com/sanjaykshetri/misinformation-at-scale)

- **Language:** Python (deep learning + Streamlit)
- **Dataset:** FakeNewsNet + verified data integrity (0 leakage)
- **Models:** DistilBERT fine-tuning (85.75% accuracy, +2.13% vs baseline)
- **Infrastructure:** Google Colab ready, Streamlit dashboards, Docker deployment
- **Comparison:** Baseline LR (83.62%) vs. DistilBERT (85.75%)
- **Key Notebook:** `notebooks/04_deep_learning_model.ipynb`
- **Deliverables:** Model cards (MODEL_CARD.md), deployment guides, ethical frameworks
- **Maps to:** `book/chapters/04-fusion-scale.qmd`
- **How to use:** Open notebook with Google Colab GPU or run locally

---

## 📊 Data (Local Copies)
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

## � Hub Book Structure (Primary Deliverable)

```
book/
├── _quarto.yml                               # Configuration (6 chapters)
├── chapters/
│   ├── 00-prologue.qmd                      # When a False Story Wins
│   ├── 01-measuring-vulnerability.qmd       # Unit 1: Behavioral Science (links to Spoke 1)
│   ├── 02-detecting-narratives.qmd          # Unit 2: NLP Models (links to Spoke 2)
│   ├── 03-modeling-spread.qmd               # Unit 3: SEIR Modeling (links to Spoke 3)
│   ├── 04-fusion-scale.qmd                  # Unit 4: Production & Ethics (links to Spoke 4)
│   └── 05-epilogue.qmd                      # Frontier Challenges & Future Directions
├── index.qmd                                 # Book homepage and TOC
├── references.bib                            # Bibliography
├── _book/                                    # Generated HTML (deployed to GitHub Pages)
├── CHAPTER_ROADMAP.md                        # Development status per unit
└── render_output.txt                         # Latest Quarto render log
```

**Live Site:** https://sanjaykshetri.github.io/tentacles-of-misinformation/

**Rendering:** Locally with Quarto v1.8.25, pre-rendered HTML committed to `_book/`

---

## 📈 Results & Models (Reference Cache)
```
results/
├── baseline_results_*.txt        # Performance summaries (from spoke repos)
├── transformer_results_*.txt
├── model_comparison.png          # Visual reference artifacts
├── roc_*.png
├── cm_*.png
├── pr_*.png
└── [other benchmark outputs]

models/
├── behavioral_model.joblib       # Serialized models (from spoke repos)
├── feature_scaler.joblib
├── hybrid_model.joblib
├── linear_svm.joblib
├── logistic_regression.joblib
└── tfidf_vectorizer.joblib
```

**Note:** These are reference caches; primary versions live in respective spoke repositories.

---

## 📚 Hub Documentation Files

```
docs/
├── README.md                     # Overview of 4-unit framework
├── CAPSTONE_PROPOSAL.md          # Original project proposal
├── THESIS_INTEGRATION.md         # How thesis maps to 4 units
├── TECHNICAL_ROADMAP.md          # Development phases
├── figures/                      # Publication-ready plots (reference)
└── architecture_diagrams/        # System architecture diagrams

REPO_STRUCTURE.md                 # This file — hub & spoke layout
DEPLOYMENT_GUIDE.md               # How to render and deploy book
book/CHAPTER_ROADMAP.md           # Per-unit development status
LICENSE                           # MIT License
README.md                         # Main project overview (portfolio hub)
```

---

## 🔧 Environment & Deployment

```
environment/
├── conda.yml                     # Conda environment spec
├── requirements.txt              # Pip dependencies
└── README.md                     # Setup instructions (for local book rendering)

.github/
└── workflows/
    └── quarto-publish.yml        # GitHub Actions: Auto-deploy to Pages on push

book/_book/                       # Generated HTML output (committed, deployed to GitHub Pages)
```

---

## 📊 Data (Local References)

| Dataset | Location | Purpose |
|---------|----------|---------|
| **FakeNewsNet** | `data/raw/fakenewsnet/` | Local copy (4 CSV files) — shared across Units 2, 3, 4 |
| **Behavioral Survey** | `behavioral_analysis/` | Survey data cache (Unit 1 origin) |
| **Models** | `models/` | Serialized model artifacts (references) |

**Primary Sources:**
- Unit 1 behavioral data → [Spoke 1 repo](https://github.com/sanjaykshetri/Misinformation-study-Masters-Thesis)
- Units 2, 3, 4 datasets → Live in respective spoke repos

---

## ✅ Repository Architecture Summary

### **Hub Responsibilities:**
- ✅ Render and publish Quarto book to GitHub Pages
- ✅ Link to and showcase 4 spoke repositories
- ✅ Provide narrative coherence across distributed work
- ✅ Serve as portfolio entry point and SEO landing page

### **Spoke Repositories (Independent):**
- ✅ Unit 1 (Spoke): Behavioral study — fully independent R project
- ✅ Unit 2 (Spoke): NLP baseline — fully independent Python project
- ✅ Unit 3 (Spoke): SEIR modeling — fully independent Python project
- ✅ Unit 4 (Spoke): Production at scale — fully independent Python project

### **Data Flow:**
```
             Published Book (GitHub Pages)
                      ↓
        [ Monograph chapters with links ]
                      ↓
    [ Embed visualizations from spoke repos ]
    [ Citations + "Learn more" & source links ]
```

---

## 📝 Maintenance Notes

- **Book Rendering:** Run `quarto render book/` locally, HTML output goes to `book/_book/`
- **Deployment:** Push main branch → GitHub Actions auto-deploys to GitHub Pages
- **Updating Units:** Edit chapters in `book/chapters/`, add links to spoke repos
- **Spoke Updates:** Each spoke repo maintains its own code, notebooks, and datasets independently
- **Hub Sync:** When spoke repos update, manually pull key findings and update chapter text/links

---

## 🔗 Quick Links to Spoke Repos

1. **Unit 1 — Behavioral Foundation**  
   → [github.com/sanjaykshetri/Misinformation-study-Masters-Thesis](https://github.com/sanjaykshetri/Misinformation-study-Masters-Thesis)

2. **Unit 2 — NLP Baselines**  
   → [github.com/sanjaykshetri/Misinformation-Detection-ML-Model2](https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2)

3. **Unit 3 — Epidemic Modeling**  
   → [github.com/sanjaykshetri/misinformation-epidemic-model](https://github.com/sanjaykshetri/misinformation-epidemic-model)

4. **Unit 4 — Production & Scale**  
   → [github.com/sanjaykshetri/misinformation-at-scale](https://github.com/sanjaykshetri/misinformation-at-scale)
````

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
