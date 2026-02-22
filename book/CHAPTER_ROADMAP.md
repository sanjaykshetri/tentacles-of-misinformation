# 📖 Book Chapter Roadmap

This document maps each Quarto book chapter to source code and data, showing which chapters are skeleton, which have draft content, and which need refinement.

---

## Chapter Status

| # | Chapter Title | Status | Linked Code/Notebooks | Key Results |
|---|---|---|---|---|
| 0 | **Prologue** — When a False Story Wins | 🟨 Draft | — | Conceptual framework |
| 1 | **Thesis as Code** — Measuring Vulnerability | 🟨 Skeleton | `behavioral_analysis/01_exploratory_analysis.ipynb` | CRT, VOI-7, CMQ-5, BSR-10 validation |
| 2 | **SPSS to Python** — Reproducible Analytics | 🟨 Skeleton | `src/data_prep.py`, `src/features.py` | Project structure, automation examples |
| 3 | **Sharing Decision** — Modeling Share Behavior | 🟡 In Progress | `behavioral_analysis/02_baseline_models.ipynb` | Logistic Regression, SVM results |
| 4 | **Pseudo-Profound Bullshit** — NLP for Linguistics | 🟡 In Progress | `behavioral_analysis/03_linguistic_features.ipynb` | BERT embeddings, TF-IDF classifier |
| 5 | **Infodemic Simulation** — Age & Susceptibility | 🟨 Skeleton | `behavioral_analysis/03_regression_models.ipynb` | Interaction modeling, risk profiles |
| 6 | **Interactive Dashboards** — Visualizing Risk | 🔴 Not Started | `dashboards/streamlit/` | Dashboard specs ready |
| 7 | **Ethics & Responsibility** — Reflective Essay | 🟨 Skeleton | — | Conceptual framework |
| 8 | **Epilogue** — My Path into Data Science | 🟨 Skeleton | `book/chapters/08-epilogue.qmd` | Portfolio framing, next directions |

---

## Legend

- 🟢 **Complete**: Narrative + code + results finalized, ready to deploy
- 🟡 **In Progress**: Partial content, being refined
- 🟨 **Skeleton**: Chapter structure and outline complete, needs content population
- 🔴 **Not Started**: Placeholder only

---

## What Each Chapter Needs

### 1️⃣ Prologue
- [x] Outline complete
- [ ] Real misinformation anecdote (1-2 paragraphs)
- [ ] Metaphor explanation (tentacles framework)
- [ ] Executive summary of book arc

### 2️⃣ Chapter 1 — Thesis as Code
- [x] Outline complete
- [ ] Link to `01_exploratory_analysis.ipynb` with embedded outputs
- [ ] Measurement validation results (Cronbach's α, normality tests)
- [ ] Path model diagram and interpretation
- [ ] Table: Descriptive statistics of CRT, VOI-7, CMQ-5, BSR-10

### 3️⃣ Chapter 2 — SPSS to Python  
- [x] Outline complete
- [ ] Before/after comparison (SPSS workflow vs. Python automation)
- [ ] Code snippets from `src/data_prep.py`
- [ ] Project structure diagram
- [ ] Video demo (optional): "How to reproduce this analysis"

### 4️⃣ Chapter 3 — Sharing Decision
- [x] Outline complete
- [ ] Dataset description (real or simulated)
- [ ] Feature importance plots from `02_baseline_models.ipynb`
- [ ] Confusion matrices and ROC curves
- [ ] Model comparison table (Accuracy, Precision, Recall, F1)
- [ ] Error analysis narrative

### 5️⃣ Chapter 4 — Pseudo-Profound Bullshit
- [x] Outline complete
- [ ] Example pseudoprofound statements
- [ ] NLP pipeline diagram
- [ ] BERT embedding visualization (t-SNE or UMAP)
- [ ] Classification results and sample predictions
- [ ] Discussion of linguistic markers

### 6️⃣ Chapter 5 — Infodemic Simulation
- [x] Outline complete
- [ ] ANOVA results table (age groups)
- [ ] Interaction plots (age vs. emotional sensitivity)
- [ ] Heatmap of risk profiles
- [ ] Simulation parameters and model description
- [ ] Time-series visualization of spread dynamics

### 7️⃣ Chapter 6 — Interactive Dashboards
- [x] Outline complete
- [ ] Dashboard mockups or screenshots
- [ ] Streamlit app setup instructions
- [ ] Tour of each visualization (network, heatmap, predictions)
- [ ] Deployment link (Streamlit Cloud)
- [ ] Code walkthrough

### 8️⃣ Chapter 7 — Ethics & Responsibility
- [x] Outline complete
- [ ] Specific case study (bias in misinformation detection, privacy risks, etc.)
- [ ] Ethical frameworks comparison table
- [ ] Links to ethics resources (Stanford IO, MIT Media Lab)
- [ ] Personal reflection on tensions

### 9️⃣ Epilogue
- [x] Outline complete
- [ ] Skills inventory table (what each chapter demonstrates)
- [ ] Career guidance narrative
- [ ] Lessons learned reflection
- [ ] Links to GitHub, portfolio, next projects

---

## Integration Checklist

For each chapter, before marking 🟢 **Complete**:

- [ ] `.qmd` file has meaningful content (not just outline)
- [ ] Code cells reference actual notebooks or scripts in repo
- [ ] All figures/tables are generated from source data or code
- [ ] Results are interpreted in context of broader project
- [ ] Links to source code and data are working
- [ ] Prose is clear for both academic and hiring manager audiences
- [ ] Chapter runs in `quarto preview` without errors
- [ ] Cross-references between chapters are correct

---

## Timeline & Priorities

### Week 1 (Priority: HIGH)
- [ ] **Ch 1 & 2**: Thesis validation + reproducibility (core portfolio strength)
- [ ] **Ch 3**: Sharing decision classification (ML foundation)

### Week 2 (Priority: HIGH)
- [ ] **Ch 4**: NLP linguistic features (differentiator)
- [ ] **Ch 7**: Ethics essay (responsibility narrative)

### Week 3 (Priority: MEDIUM)
- [ ] **Ch 5**: Infodemic simulation (complexity + visualization)
- [ ] **Ch 6**: Dashboards setup (decision support)

### Week 4 (Priority: POLISH)
- [ ] **Prologue & Epilogue**: Narrative framing
- [ ] Cross-linking and consistency
- [ ] Book-wide review and edits

---

## How to Reference Notebooks in Chapters

Use Quarto's inline code execution:

```markdown
# Results from baseline models

See the analysis in [baseline models notebook](#).

#| include: false
import json
with open('../results/baseline_results_20260116_164011.txt') as f:
    results = f.read()

{r} results
```

Or embed results directly:

```markdown
{{< include ../behavioral_analysis/notebooks/02_baseline_models.ipynb#fig-roc >}}
```

---

## Questions & Support

- **Build issues**: See `DEPLOYMENT_GUIDE.md`
- **Chapter outline help**: Review Perplexity.ai conversation for expanded brief
- **Data access**: All datasets in `data/` directory with README
- **Code questions**: Check corresponding `.ipynb` or `.py` files for implementation

---

**Last updated**: February 22, 2026  
**Book version**: 0.1 (skeleton)  
**Target launch**: March 2026
