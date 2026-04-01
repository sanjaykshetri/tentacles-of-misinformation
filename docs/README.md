# Documentation

## 4-Unit Research Framework

This project is organized around four research units, corresponding to chapters in the research monograph:

### **Unit 1: Measuring Vulnerability** (Behavioral Science)
- IRB-approved behavioral survey data (N=194)
- Cognitive features: CRT, conspiracy mentality, bull*** receptivity
- Statistical validation: Cronbach's α, factor analysis, path models
- Files: `docs/THESIS_INTEGRATION.md`, `behavioral_analysis/README.md`

### **Unit 2: Detecting Narratives** (NLP Pipeline)
- FakeNewsNet, LIAR dataset, PolitiFact corpus
- Models: TF-IDF → Linguistic features → RoBERTa transformers
- Evaluation: ROC-AUC, confusion matrices, cross-domain generalization
- Files: `nlp_models/README.md`, `results/`

### **Unit 3: Modeling Spread** (Epidemiology)
- SEIR model applied to misinformation cascades
- Parameter calibration from behavioral + NLP data
- Sensitivity analysis: β (transmission), γ (recovery)
- Interference scenarios and policy recommendations
- Files: `fusion_models/README.md`, `fusion_models/experiments/`

### **Unit 4: Fusion & Scale** (Production & Ethics)
- Multimodal ensemble combining behavioral + content signals
- Dashboard design and interactive visualizations
- Ethical framework: privacy, fairness, transparency, accountability
- Deployment architecture and monitoring
- Files: `dashboards/README.md`, `DEPLOYMENT_GUIDE.md`

---

## Contents

- **figures/** - Publication-ready figures and plots (generate from `results/`)
- **architecture_diagrams/** - System and model architecture diagrams
- **THESIS_INTEGRATION.md** - How thesis research maps to 4-unit framework
- **TECHNICAL_ROADMAP.md** - Development phases and dependencies
- **CAPSTONE_PROPOSAL.md** - Original project proposal

---

## Key Data & Results

### Behavioral Data
- Source: Master's thesis IRB-approved survey (N=194)
- Location: `data/raw/`, `data/processed/`
- Features: CRT, NFC, conspiracy beliefs, BS receptivity, verification behavior

### NLP Results
- Location: `results/` (PNG visualizations)
- Files: `cm_*.png` (confusion matrices), `roc_*.png` (ROC curves), `model_comparison.png`
- Notebooks: `nlp_models/notebooks/`, `behavioral_analysis/notebooks/`

### Model Performance
- Unit 2 (NLP): 82-87% accuracy across models, 0.85-0.87 ROC-AUC
- Unit 1 (Behavioral): CRT significantly predicts verification behavior (β=0.149, p=.031)
- Unit 3 (Epidemiology): R₀=0.176 baseline, 14.3% intervention effectiven

ess
- Unit 4 (Fusion): 84-85% ensemble accuracy on test set

---

## Visualization Guide

### Unit 1 - Behavioral Features
- Feature distributions and correlation heatmaps
- Demographic breakdowns and risk profiles
- Path models and factor loadings

### Unit 2 - NLP Performance  
- Confusion matrices and ROC curves (in `results/`)
- Feature importance plots (TF-IDF top words)
- Error analysis by misinformation type
- Model comparison table

### Unit 3 - Spread Dynamics
- SEIR time-series trajectories
- Sensitivity heatmaps (β vs γ)
- Population heterogeneity effects
- Intervention scenario comparisons

### Unit 4 - System Architecture
- Data pipeline diagram (ingest → preprocess → ensemble → API)
- Dashboard component mockups
- Ethical governance framework matrix
- Monitoring and drift detection plots

---

## How to Reference

All figures should be referenced in the Quarto book (`book/chapters/`).

## Updating Documentation

As the project evolves:
1. Update relevant chapter in `book/chapters/`
2. Save new figures in `docs/figures/`
3. Update architecture diagrams if major changes occur
4. Rebuild the book with Quarto

## Tools

- Matplotlib / Seaborn for publication-quality plots
- Graphviz / Draw.io for architecture diagrams
- Quarto for document generation
