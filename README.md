# 🐙 The Tentacles of Misinformation

A **research monograph and portfolio showcase** integrating four specialized repositories on misinformation—from behavioral science foundations through NLP detection, epidemiological modeling, and production-scale systems.

This book assembles self-contained research projects into a cohesive narrative: **"What is misinformation? How can we detect it? How does it spread? How do we operationalize solutions?"**

---

## 📖 The 4-Unit Research Framework

Each book unit corresponds to a dedicated GitHub repository:

### **Unit 1 — Measuring Vulnerability** (Behavioral-Cognitive Foundation)
**Source Repo:** [Misinformation-study-Masters-Thesis](https://github.com/sanjaykshetri/Misinformation-study-Masters-Thesis)

- **What:** IRB-approved behavioral study with N=194
- **Key Finding:** Cognitive Reflection Test (CRT) predicts information verification (β=0.149, p=.031)
- **Methods:** Mediation analysis, SEM, Qualtrics survey design
- **Deliverables:** Thesis PDF, data dictionary, reproducible R analysis
- **Audience:** Behavioral scientists, psychologists, researchers interested in cognitive foundations

---

### **Unit 2 — Detecting Narratives** (NLP & Machine Learning)
**Source Repos:** 
- [Misinformation-Detection-ML-Model2](https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2)  
- [misinformation-at-scale](https://github.com/sanjaykshetri/misinformation-at-scale)

- **What:** Evolution from baselines (TF-IDF + Logistic Regression) to deep learning (DistilBERT)
- **Key Finding:** Baseline accuracy 83.62% → DistilBERT accuracy 85.75% (+16% recall)
- **Methods:** FakeNewsNet dataset, statistical validation, data leakage prevention
- **Datasets:** 23,194 real fact-checked articles (PolitiFact + GossipCop)
- **Deliverables:** Notebooks, model cards, ethical considerations
- **Audience:** Data scientists, NLP practitioners, model developers

---

### **Unit 3 — Modeling Spread** (Epidemiological Framework)
**Source Repo:** [misinformation-epidemic-model](https://github.com/sanjaykshetri/misinformation-epidemic-model)

- **What:** SEIR compartmental model applied to misinformation cascades
- **Key Finding:** Calibrated parameters (β=0.0153, R₀=0.176) from FakeNewsNet data
- **Methods:** ODE-based simulation, sensitivity analysis, intervention scenarios
- **Visualizations:** Publication-quality heatmaps, confidence bands, ensemble trajectories
- **Deliverables:** Python module, Jupyter notebooks, academic documentation
- **Audience:** Epidemiologists, systems modelers, policy researchers

---

### **Unit 4 — Fusion & Scale** (Production & Ethics)
**Source Repo:** [misinformation-at-scale](https://github.com/sanjaykshetri/misinformation-at-scale)

- **What:** Production-ready misinformation detection system
- **Key Finding:** Real-world 84-86% accuracy with excellent generalization (2.5% train-val gap)
- **Methods:** Baseline + deep learning ensemble, Streamlit dashboards, Docker deployment
- **Infrastructure:** Google Colab, FastAPI, ethical frameworks
- **Deliverables:** Deployment guides, model cards, interactive dashboards
- **Audience:** Product teams, platform engineers, policy makers, practitioners

---

## 🏗️ Portfolio Architecture

```
tentacles-of-misinformation/  ← This monograph (hub)
├── book/                     ← Quarto-rendered research narrative
│   ├── chapters/01-04-*.qmd  ← Each chapter links to external repo
│   └── _book/                ← Published HTML + site
│
├── Unit 1 (External) ←→ Misinformation-study-Masters-Thesis
├── Unit 2 (External) ←→ Misinformation-Detection-ML-Model2
├── Unit 3 (External) ←→ misinformation-epidemic-model
└── Unit 4 (External) ←→ misinformation-at-scale
```

Each book chapter **showcases, summarizes, and links to** its corresponding repository. The book serves as a portfolio piece that demonstrates:
- Interdisciplinary research (psychology + data science + epidemiology)
- End-to-end project execution (theory → implementation → deployment)
- Academic rigor (publications, ethics, reproducibility)
- Production readiness (dashboards, APIs, documentation)

---

## 🌐 Live Site

**Published Monograph:** https://sanjaykshetri.github.io/tentacles-of-misinformation/

The book is automatically deployed to GitHub Pages whenever changes are pushed to the `main` branch.

---

## 📚 Quick Navigation

| Chapter | Topic | External Repo | Time |
|---------|-------|---------------|------|
| [Prologue](https://sanjaykshetri.github.io/tentacles-of-misinformation/chapters/00-prologue.html) | Overview & framing | — | 5 min |
| [Unit 1](https://sanjaykshetri.github.io/tentacles-of-misinformation/chapters/01-measuring-vulnerability.html) | Behavioral Science | [Masters Thesis](https://github.com/sanjaykshetri/Misinformation-study-Masters-Thesis) | 20 min |
| [Unit 2](https://sanjaykshetri.github.io/tentacles-of-misinformation/chapters/02-detecting-narratives.html) | NLP & ML | [Detection Repos](https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2) | 25 min |
| [Unit 3](https://sanjaykshetri.github.io/tentacles-of-misinformation/chapters/03-modeling-spread.html) | Epidemiology | [Epidemic Model](https://github.com/sanjaykshetri/misinformation-epidemic-model) | 20 min |
| [Unit 4](https://sanjaykshetri.github.io/tentacles-of-misinformation/chapters/04-fusion-scale.html) | Production & Ethics | [At Scale](https://github.com/sanjaykshetri/misinformation-at-scale) | 25 min |
| [Epilogue](https://sanjaykshetri.github.io/tentacles-of-misinformation/chapters/05-epilogue.html) | Frontier Challenges | — | 10 min |

---

## 🎯 Use Cases

### For Hiring Managers
See how research ideas progress from concept → theory → implementation → production:
- **Unit 1**: Behavioral hypothesis testing and statistical rigor
- **Unit 2**: ML engineering, model cards, ethical considerations  
- **Unit 3**: Complex systems modeling and analysis
- **Unit 4**: Full-stack deployment and dashboards

### For Researchers
Reference implementations for:
- Behavioral survey design and analysis (Unit 1)
- NLP baseline → state-of-the-art progression (Unit 2)
- Calibrating epidemiological models with real data (Unit 3)
- Responsible AI deployment (Unit 4)

### For Policy Makers
Evidence for intervention effectiveness:
- Individual-level cognitive interventions (Unit 1)
- Automated detection pipelines (Unit 2)
- Population-level mitigation strategies (Unit 3: 14.3% burden reduction)
- Ethical deployment frameworks (Unit 4)

---

## 🚀 Getting Started

### View the Published Book
Simply visit: **https://sanjaykshetri.github.io/tentacles-of-misinformation/**

### Explore Individual Repos
Each repository is self-contained and ready to use:

```bash
# Unit 1: Behavioral Science
git clone https://github.com/sanjaykshetri/Misinformation-study-Masters-Thesis.git
cd Misinformation-study-Masters-Thesis
# See data_analysis1.R for reproducible analysis

# Unit 2: NLP Detection
git clone https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2.git
cd Misinformation-Detection-ML-Model2
jupyter notebook misinformation_analysis.ipynb

# Unit 3: Epidemic Model  
git clone https://github.com/sanjaykshetri/misinformation-epidemic-model.git
cd misinformation-epidemic-model
jupyter notebook notebooks/quick_start_academic.ipynb

# Unit 4: At Scale
git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
cd misinformation-at-scale
python run_complete_training.py  # Or open in Google Colab
```

### Reproducible Monograph
Build the book locally:
```bash
cd book
quarto render --no-execute
```
Output: `book/_book/index.html` (open in browser)

---

## 📋 Project Contents

**This Repository (Hub):**
- Quarto-based research monograph linking all 4 units
- REPO_STRUCTURE.md — Architecture documentation
- CHAPTER_ROADMAP.md — Development status and priorities
- DEPLOYMENT_GUIDE.md — How to render and publish the book

**External Repositories (Spokes):**
| Repo | Purpose | Status |
|------|---------|--------|
| Misinformation-study-Masters-Thesis | Behavioral foundations | ✅ Complete (published thesis) |
| Misinformation-Detection-ML-Model2 | NLP baseline models | ✅ Complete (83.62% accuracy) |
| misinformation-at-scale | Deep learning + dashboards | ✅ Complete (85.75% accuracy) |
| misinformation-epidemic-model | SEIR simulation | ✅ Complete (46 unit tests) |

---

## 🛠️ Technologies

**Across All Units:**
- Python 3.9+, R 4.0+, JavaScript/HTML
- Data: Qualtrics surveys, FakeNewsNet dataset (23K+ articles)
- ML: scikit-learn, PyTorch, transformers, XGBoost
- Math: scipy.integrate (ODE), statsmodels, lavaan (SEM)
- Deployment: Quarto, Streamlit, Docker, GitHub Pages
- Infrastructure: GitHub Actions CI/CD, Google Colab

---

## 📖 Reading Guide

**For a quick overview (30 minutes):**
1. Read Prologue
2. Skim Unit summaries (each ~5 min)
3. Review Epilogue for frontier challenges

**For deep dives (2-3 hours):**
1. Pick a unit that interests you
2. Read the chapter fully (25 min)
3. Clone the external repo
4. Explore the notebooks and run code

**For reproducibility:**
- Each unit is fully self-contained in its external repo
- RUN guides in each README
- All code is open source (MIT license)

---

## 👤 About

**Sanjay Kumar Chhetri**  
Data Scientist | Behavioral Researcher | Educator

- **Master's:** Psychology (Misinformation vulnerability)
- **Capstone:** Data Science (Springboard)
- **Research Interests:** Behavioral science × computational methods, responsible AI, population-level interventions

---

## 📄 License

MIT License - All code is open source.  
Data use follows original dataset licenses (FakeNewsNet, Qualtrics ethical guidelines, IRB approval for behavioral study).

---

## 📧 Contact

**Questions or Collaboration?**
- GitHub: [@sanjaykshetri](https://github.com/sanjaykshetri)
- Email: Available via GitHub profile

---

**Last Updated:** April 1, 2026  
**Book Version:** 2.0 (4-unit architecture with external repo integration)  
**Status:** Live and continuously updated
