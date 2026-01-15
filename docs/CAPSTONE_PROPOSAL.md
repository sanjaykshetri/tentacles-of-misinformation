# 🎓 Springboard Final Capstone Proposal

## Title

### **The Tentacles of Misinformation: Predicting Susceptibility Using Behavioral Data and NLP**

---

## 1. Problem Statement

Online misinformation poses serious risks to public health, democracy, and social cohesion.
While most machine learning approaches focus on detecting whether content is false, far fewer attempt to model **who is most vulnerable to believing or spreading misinformation**, and why.

This project aims to integrate:

* behavioral and cognitive data from human subjects research, and
* large-scale text-based misinformation datasets

to build predictive models that estimate **misinformation susceptibility as a function of both individual traits and content characteristics**.

This approach reflects real-world trust & safety and policy challenges, where both **user profiles and narrative framing** influence misinformation impact.

---

## 2. Data Sources

### A. Behavioral Dataset (Primary)

From IRB-approved master's thesis research (N = 194):

**Features include:**

* Cognitive Reflection Test (CRT)
* Rational vs Intuitive decision style
* Conspiracy mentality
* Bullshit receptivity
* Self-reported verification behavior (proxy for misinformation vulnerability)

**Data collected via:**

* Qualtrics survey
* CloudResearch recruitment
* De-identified and ethically approved for analysis

---

### B. Text-Based Misinformation Datasets

Public NLP datasets such as:

* FakeNewsNet
* LIAR dataset
* PolitiFact labeled statements
* Social media misinformation corpora

**Text features will include:**

* TF-IDF baselines
* Transformer embeddings (e.g., SBERT)
* Emotional framing metrics
* Topic modeling clusters

---

## 3. Project Objectives

1. Identify behavioral predictors of misinformation vulnerability
2. Train NLP models to classify and characterize misinformation narratives
3. Build fusion models combining:
   * human cognitive features
   * content-level features
4. Predict which types of users are most susceptible to which types of misinformation
5. Build interactive dashboards to communicate findings

---

## 4. Methods & Modeling Approach

### Behavioral Modeling

* Regression models (logistic, linear)
* Tree-based models (Random Forest, XGBoost)
* Feature importance analysis (SHAP values)

### NLP Modeling

* Text preprocessing pipelines
* Sentence embeddings (transformers: SBERT, RoBERTa)
* Classification models for misinformation detection
* Narrative clustering via topic models

### Fusion Models

Multimodal ML approaches combining:

* Behavioral vectors
* Narrative embeddings

**Models may include:**

* Gradient boosting
* Neural networks with attention mechanisms
* Ensemble approaches

**Evaluation metrics:**

* ROC-AUC
* F1-score
* Calibration curves
* Cross-validation strategies

---

## 5. Visualization & Dashboard Layer

Interactive dashboards will show:

* Behavioral risk profiles
* Narrative vulnerability maps
* Predicted susceptibility by topic
* Model confidence and uncertainty

**Tools:**

* Streamlit for data science dashboards
* Power BI for executive-style reports
* Matplotlib / Seaborn for publication figures

**Goal:**
Translate technical outputs into decision-support tools for researchers and policymakers.

---

## 6. Deliverables

### Technical

* ✅ Modular ML pipelines (reproducible, tested)
* ✅ Jupyter notebooks with full EDA and modeling
* ✅ Clean, documented GitHub repository
* ✅ Trained models (stored with Git LFS)
* ✅ Unit tests and validation scripts

### Communication

* ✅ Interactive dashboards (Streamlit MVP + Power BI concept)
* ✅ Quarto research book: *The Tentacles of Misinformation*
* ✅ Public website deployment
* ✅ Publication-ready figures and tables

### Research

* ✅ Journal manuscript based on behavioral study
* ✅ Replicable methodology documentation
* ✅ Ethical considerations and IRB compliance summary

---

## 7. Why This Project Matters

This project addresses:

* **AI ethics** — Understanding user vulnerability, not just content detection
* **Trust & Safety** — Real-world problems for platforms and policy
* **Digital literacy** — Behavioral foundations for misinformation resistance
* **Platform responsibility** — User-centric vs. content-centric approaches

It bridges:

* **Psychology** — cognitive assessment and behavioral predictors
* **Data Science** — statistical modeling and feature engineering
* **Machine Learning** — NLP and fusion architectures
* **Public Policy** — actionable insights for decision-makers

This reflects real-world applied data science challenges rather than purely technical benchmarks.

---

## 8. Tools & Technologies

**Languages & Frameworks:**
* Python 3.11 (core development)
  * pandas, numpy, scipy — data manipulation
  * scikit-learn — classical ML
  * PyTorch — neural networks
  * transformers (HuggingFace) — NLP models
  * statsmodels — statistical testing

**Dashboards & Visualization:**
* Streamlit — interactive web dashboards
* Plotly — interactive plots
* Power BI — executive dashboards (optional)

**Development & Deployment:**
* Jupyter / JupyterLab — notebooks
* Git / GitHub — version control
* Quarto — research publishing

**Infrastructure:**
* Conda / pip — environment management
* pytest — unit testing
* Git LFS — large model storage

---

## 9. Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Small behavioral dataset (N=194) | Augment with simulation; focus on modeling methodology and interpretability rather than scale |
| Multimodal complexity | Build staged pipelines: behavioral → NLP → fusion; validate each layer independently |
| Time constraints | Prioritize behavioral + NLP separately; fusion can be iterative |
| Data privacy concerns | Use only anonymized, IRB-approved behavioral data; external datasets are public and licensed |
| Model interpretability | Emphasize SHAP values, feature importance, and ablation studies over black-box accuracy |

---

## 10. Success Criteria

* ✅ Working predictive models for behavioral susceptibility
* ✅ End-to-end NLP classification pipeline
* ✅ Multimodal fusion architecture with clear performance gains
* ✅ Interactive dashboard demonstrating insights
* ✅ Public research website/book
* ✅ Strong capstone rubric coverage (all sections)
* ✅ Interview-ready narrative and GitHub portfolio

---

## 11. Timeline & Milestones

### Phase 1: Behavioral Analysis (Weeks 1-3)
* Load and explore thesis data
* Feature engineering and statistical testing
* Regression and tree-based models
* Feature importance analysis

### Phase 2: NLP Pipeline (Weeks 4-6)
* Download and preprocess misinformation datasets
* Baseline text classification models
* Transformer-based classifiers
* Evaluation and error analysis

### Phase 3: Fusion & Integration (Weeks 7-9)
* Feature alignment between modalities
* Multimodal architecture design
* Model training and hyperparameter tuning
* Comparative performance analysis

### Phase 4: Dashboards & Communication (Weeks 10-12)
* Streamlit dashboard development
* Quarto book writing
* Final visualizations and publication prep
* Capstone submission and presentation

---

## 12. Why This Capstone Is Strategic

This capstone positions you as:

* ✔️ **Behavioral data scientist** — research background in human cognition
* ✔️ **NLP practitioner** — transformer models and text classification
* ✔️ **Applied ML engineer** — multimodal fusion and real-world problems
* ✔️ **Research communicator** — documentation, dashboards, publications

It directly aligns with:

* Your Master's thesis (bridges academic work to industry)
* Your misinformation research interests
* Your education and psychology background
* **$150k+ roles** in trust & safety, applied AI, and research science

You are not presenting another Kaggle classifier.
You are presenting a **research platform** that solves real-world problems.

---

## 13. Author & Contact

**Sanjay Kumar Chhetri**  
Data Scientist | Behavioral Researcher | Educator  
GitHub: https://github.com/sanjaykshetri  
Website: https://mathwithmeditation.com

---

**IRB & Ethics:**
All behavioral data used in this project is anonymized and approved for analysis under IRB protocol (Montclair State University). Public misinformation datasets are used in accordance with their respective licenses.
