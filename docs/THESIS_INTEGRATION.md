# Thesis Integration Guide

## Your Master's Thesis

**Title:** Cognitive Vulnerability to Misinformation  
**Institution:** Montclair State University, 2023  
**Ethics:** IRB-approved  
**Sample:** N=194 with behavioral assessment

Your thesis is the **foundation** of this research project. It provides the behavioral data, theoretical framework, and initial findings that are now extended through NLP and epidemiological modeling.

---

## How Thesis Research Maps to 4 Units

### **Unit 1: Measuring Vulnerability** ← Thesis Core
Your thesis research becomes Unit 1, with enhancements:

**Original Thesis Content:**
- Behavioral survey design and instrument validation
- Cognitive features: CRT, NFC, conspiracy mentality, BS receptivity
- Statistical tests: factor analysis, correlations, regression
- Key finding: CRT correlates with verification behavior (β=0.149, p=.031)

**Unit 1 Enhancements:**
- Reproduced in Python (statsmodels, scipy) for reproducibility
- Publication-quality visualizations (seaborn, matplotlib)
- Expanded feature engineering and composite scoring
- Path models and mediation analysis
- Distribution analysis and demographic breakdowns
- **See:** `book/chapters/01-measuring-vulnerability.qmd`

---

### **Unit 2: Detecting Narratives** ← Thesis Extension
Use behavioral insights to improve NLP models:

**Connection to Thesis:**
- Thesis identified that *individual differences* drive susceptibility
- Unit 2 asks: What *content characteristics* interact with these differences?
- Bridge: Use behavioral profiles to improve text classification

**How It Works:**
- Train NLP classifiers on FakeNewsNet (which types do vulnerable people share?)
- High performers: RoBERTa (87% AUC)
- Analyze errors: What content fools both people and models?
- **See:** `book/chapters/02-detecting-narratives.qmd`, `nlp_models/README.md`

---

### **Unit 3: Modeling Spread** ← Thesis Application
Apply behavioral insights to population-level dynamics:

**Connection to Thesis:**
- Thesis measured *individual susceptibility*
- Unit 3 scales up: If X% of population has high CRT, what infection rate emerges?
- SEIR model adapts epidemiology to misinformation

**How It Works:**
- Calibrate transmission rate (β) from behavioral correlations
- Calibrate recovery rate (γ) from fact-checking dynamics
- Simulate: Heterogeneous population with varying CRT, NFC, conspiracy beliefs
- Intervention targets: education, friction, content labels
- **See:** `book/chapters/03-modeling-spread.qmd`, `fusion_models/experiments/`

---

### **Unit 4: Fusion & Scale** ← Thesis Deployment
Build production systems using all insights:

**Connection to Thesis:**
- Thesis: Behavioral features predict susceptibility *in sample*
- Unit 4: Deploy hybrid model (behavioral + NLP) at scale
- Ethical governance: How to use individual-level predictions responsibly?

**How It Works:**
- Ensemble: Combine behavioral signals + content signals + user history
- Dashboards: Interactive visualization for policy and product decisions
- Ethics framework: Address privacy, fairness, consent, accountability
- Monitoring: Track model drift, demographic parity, false positive rates
- **See:** `book/chapters/04-fusion-scale.qmd`, `dashboards/README.md`

---

## Source Code Organization

| Unit | Primary Directory | Key Files |
|------|-------------------|-----------|
| **Unit 1** | `behavioral_analysis/` | `notebooks/01_exploratory_analysis.ipynb` |
| **Unit 2** | `nlp_models/` | `notebooks/02_baseline_models.ipynb` |
| **Unit 3** | `fusion_models/experiments/` | SEIR calibration & simulation |
| **Unit 4** | `dashboards/` + `src/` | Streamlit apps, API serving |

---

## Key Behavioral Features (from thesis)

- **Cognitive Reflection Test (CRT)** — Ability to override intuitive (wrong) answers
- **Need for Cognition (NFC)** — Enjoyment of effortful thinking
- **Conspiracy Mentality** — Tendency to attribute events to hidden conspiracies
- **Bullshit Receptivity** — Susceptibility to pseudo-profound nonsense
- **Rational vs Intuitive Style** — Decision-making approach
- **Verification Behavior** — Self-reported fact-checking tendency

---

## Next Steps

- [ ] Confirm thesis data access and IRB status
- [ ] Validate Python pipeline reproduces original statistical findings
- [ ] Cross-validate behavioral features with NLP classifiers
- [ ] Calibrate SEIR model with empirical estimates
- [ ] Build dashboard with ethical safeguards
- [ ] Write and submit paper drafts
- [ ] Publish research monograph on GitHub Pages

---

**Thesis Citation:**
```bibtex
@mastersthesis{chhetri2023,
  title={Cognitive Vulnerability to Misinformation},
  author={Chhetri, Sanjay Kumar},
  school={Montclair State University},
  year={2023}
}
```

**Current Project Citation:**
```bibtex
@online{chhetri2026,
  title={The Tentacles of Misinformation: Behavioral, NLP, and Epidemiological Perspectives},
  author={Chhetri, Sanjay Kumar},
  url={https://sanjaykshetri.github.io/tentacles-of-misinformation/},
  year={2026}
}
```
