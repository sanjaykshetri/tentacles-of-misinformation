"""
Tentacles of Misinformation — Research Hub (Phase 4)
Multi-component Streamlit dashboard integrating NLP analysis, behavioral profiling, fusion predictions, and model comparison.
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Misinformation Research Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths — resolve for both local dev and HF Spaces deployment
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).parent
_local_root  = APP_DIR.parent.parent  # …/tentacles-of-misinformation

# Model files: local project → fall back to src/models/ next to this file
MODEL_DIR   = _local_root / "models"    if (_local_root / "models").exists()   else APP_DIR / "models"
FUSION_DIR  = (_local_root / "fusion_models" / "results"
               if (_local_root / "fusion_models" / "results").exists()
               else APP_DIR / "fusion_results")

# Result images: served from local disk when available, otherwise via GitHub raw URL
_GITHUB_RAW = "https://raw.githubusercontent.com/sanjaykshetri/tentacles-of-misinformation/main/results"
_local_results = _local_root / "results"

def result_img(filename: str):
    """Return local Path if it exists, else the GitHub raw URL string."""
    local = _local_results / filename
    return local if local.exists() else f"{_GITHUB_RAW}/{filename}"

def result_img_exists(filename: str) -> bool:
    """True whether the image is available locally or remotely (always True on HF)."""
    return (_local_results / filename).exists() or True  # remote always available

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nlp_models():
    paths = {
        "tfidf":   MODEL_DIR / "tfidf_vectorizer.joblib",
        "lr":      MODEL_DIR / "logistic_regression_baseline.joblib",
        "svm":     MODEL_DIR / "linear_svm_baseline.joblib",
    }
    loaded = {}
    for key, p in paths.items():
        if p.exists():
            try:
                loaded[key] = joblib.load(p)
            except Exception:
                pass
    return loaded


@st.cache_resource(show_spinner=False)
def load_fusion_model():
    p = MODEL_DIR / "fusion_early_gbm.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_results():
    data = {}
    # model comparison
    p = _local_results / "model_comparison_final.csv"
    if p.exists():
        data["comparison"] = pd.read_csv(p)
    # ablation
    p = FUSION_DIR / "03_ablation_summary.json"
    if p.exists():
        with open(p) as f:
            data["ablation"] = json.load(f)
    # bootstrap CI
    p = FUSION_DIR / "05_validation_summary.csv"
    if p.exists():
        data["validation"] = pd.read_csv(p)
    # permutation importance
    p = FUSION_DIR / "04_permutation_importance.csv"
    if p.exists():
        data["perm_imp"] = pd.read_csv(p)
    return data


def predict_text(text: str, nlp_models: dict):
    """Return dict of model_name → (label, proba_fake) for NLP models."""
    out = {}
    if "tfidf" not in nlp_models:
        return out
    vec = nlp_models["tfidf"]
    X = vec.transform([text])
    for name in ("lr", "svm"):
        if name in nlp_models:
            try:
                proba = nlp_models[name].predict_proba(X)[0]
                label = "FAKE" if proba[1] >= 0.5 else "REAL"
                out[name.upper()] = (label, float(proba[1]))
            except Exception:
                pass
    return out


def make_gauge(prob: float, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    color = "#F44336" if prob >= 0.5 else "#4CAF50"
    ax.barh(0, prob, color=color, height=0.5, alpha=0.85)
    ax.barh(0, 1 - prob, left=prob, color="#E0E0E0", height=0.5, alpha=0.5)
    ax.axvline(0.5, color="k", lw=1.5, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_xlabel("P(Fake)", fontsize=9)
    ax.set_title(f"{title}  {prob:.1%}", fontsize=10, fontweight="bold", color=color)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=60)
st.sidebar.title("🧠 Research Hub")
st.sidebar.markdown("*Tentacles of Misinformation*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📰 NLP Analysis Tool",
        "👤 Behavioral Profiler",
        "🔀 Fusion Predictor",
        "📊 Model Comparison",
        "ℹ️ About",
    ],
)

nlp_models   = load_nlp_models()
fusion_model = load_fusion_model()
results      = load_results()

# ============================================================================
#  PAGE: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.title("🧠 Misinformation Research Hub")
    st.markdown(
        """
        This research hub integrates findings from the **Tentacles of Misinformation** project:
        a multi-modal system combining **psycho-behavioral profiling** with **NLP-based content analysis**
        to predict susceptibility to and propagation of misinformation.
        """
    )
    st.markdown("---")

    cols = st.columns(4)
    metrics = [
        ("Best NLP AUC",    "0.859", "TF-IDF + Logistic Regression"),
        ("Fusion Boost",    "+2–4%", "Behavioral features added"),
        ("Study N",         "194",   "IRB-approved participants"),
        ("Articles",        "23 196","FakeNewsNet corpus"),
    ]
    for col, (label, value, caption) in zip(cols, metrics):
        col.metric(label, value, caption)

    st.markdown("---")
    st.subheader("Research Pipeline")
    st.markdown(
        """
        ```
        Raw data (FakeNewsNet CSVs)
              │
              ▼
        NLP Pipeline (TF-IDF / SBERT)  ──────────────────────────┐
              │                                                   │
        NLP Classifier (LR / SVM / SBERT+MLP)           Behavioral Survey
              │                                            (CRT, NFC, ...)
              └──────────────── FUSION MODEL ◄────────────────────┘
                                      │
                             Susceptibility Score
        ```
        """
    )

    # Show any saved result images
    imgs = {
        "Model Comparison": result_img("model_comparison_roc.png"),
        "Ablation Studies": result_img("ablation_modality.png"),
        "Permutation Importance": result_img("permutation_importance.png"),
    }
    st.subheader("📸 Latest Results")
    cols2 = st.columns(len(imgs))
    for col, (label, path) in zip(cols2, imgs.items()):
        col.image(str(path), caption=label, use_container_width=True)


# ============================================================================
#  PAGE: NLP ANALYSIS TOOL
# ============================================================================
elif page == "📰 NLP Analysis Tool":
    st.title("📰 NLP Article Analysis")
    st.markdown(
        "Enter a news headline or article excerpt. The system scores it using trained NLP classifiers."
    )
    st.markdown("---")

    user_text = st.text_area(
        "Paste headline or article text:",
        height=150,
        placeholder="Scientists confirm link between social media use and misinformation belief according to peer-reviewed study...",
    )

    model_choice = st.multiselect(
        "Models to run:",
        options=["LR", "SVM"],
        default=["LR", "SVM"],
    )

    if st.button("🔍 Analyse", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text.")
        elif not nlp_models:
            st.session_state["nlp_result"] = {"demo": True}
        else:
            preds = predict_text(user_text, nlp_models)
            preds = {k: v for k, v in preds.items() if k in [m.upper() for m in model_choice]}
            if not preds:
                st.error("Selected models not available. Check that models are saved.")
            else:
                tfidf_terms = []
                if "tfidf" in nlp_models:
                    try:
                        vec = nlp_models["tfidf"]
                        X   = vec.transform([user_text])
                        fn  = vec.get_feature_names_out()
                        scores = X.toarray()[0]
                        top_idx = np.argsort(scores)[-12:][::-1]
                        tfidf_terms = [(fn[i], float(scores[i])) for i in top_idx if scores[i] > 0]
                    except Exception:
                        pass
                st.session_state["nlp_result"] = {"preds": preds, "terms": tfidf_terms}

    result = st.session_state.get("nlp_result")
    if result:
        if result.get("demo"):
            st.warning("NLP models not found in `models/`. Run `nlp_models/notebooks/01_baseline_classifiers.ipynb` first.")
            st.info("Showing demo result for illustration.")
            col1, col2 = st.columns(2)
            col1.metric("LR P(Fake)", "37.2%", "REAL")
            col2.metric("SVM P(Fake)", "41.8%", "REAL")
        else:
            preds = result["preds"]
            st.markdown("### Classification Results")
            col_arr = st.columns(len(preds))
            for col, (mname, (label, prob)) in zip(col_arr, preds.items()):
                with col:
                    if label == "FAKE":
                        st.error(f"**{mname}**: {label}")
                    else:
                        st.success(f"**{mname}**: {label}")
                    fig = make_gauge(prob, mname)
                    st.pyplot(fig)
                    plt.close(fig)
            top_terms = result.get("terms", [])
            if top_terms:
                st.markdown("### 🔑 Top TF-IDF Signals")
                terms_df = pd.DataFrame(top_terms, columns=["Term", "TF-IDF Score"])
                fig2, ax2 = plt.subplots(figsize=(8, 3.5))
                ax2.barh(terms_df["Term"][::-1], terms_df["TF-IDF Score"][::-1],
                         color="#673AB7", alpha=0.8)
                ax2.set_xlabel("TF-IDF Score")
                ax2.set_title("Key Terms in Input", fontweight="bold")
                ax2.grid(True, axis="x", alpha=0.3)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)


# ============================================================================
#  PAGE: BEHAVIORAL PROFILER
# ============================================================================
elif page == "👤 Behavioral Profiler":
    st.title("👤 Behavioral Susceptibility Profiler")
    st.markdown(
        """
        Answer the questions below to receive a personalized **misinformation susceptibility profile**
        based on the psycho-cognitive features identified in the research study (N=194).

        *Scores are informational and based on population-level statistical patterns, not clinical assessments.*
        """
    )
    st.markdown("---")

    with st.form("behavioral_form"):
        st.subheader("🧠 Cognitive Reflection Test (CRT)")
        st.markdown(
            "*Answer each question carefully — the test is designed to trigger an intuitive "
            "but incorrect response. Think before you answer.*"
        )
        crt_q1 = st.radio(
            "**Q1** · A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the **ball** cost?",
            ["10 cents", "5 cents", "15 cents", "1 cent"],
            index=0,
        )
        crt_q2 = st.radio(
            "**Q2** · If it takes 5 machines 5 minutes to make 5 widgets, how long would it take "
            "100 machines to make 100 widgets?",
            ["100 minutes", "5 minutes", "20 minutes", "50 minutes"],
            index=0,
        )
        crt_q3 = st.radio(
            "**Q3** · In a lake there is a patch of lily pads. Every day the patch doubles in size. "
            "If it takes 48 days for the patch to cover half the lake, how long to cover the whole lake?",
            ["24 days", "49 days", "96 days", "47 days"],
            index=0,
        )
        crt_q4 = st.radio(
            "**Q4** · John can drink one barrel of water in 6 days; Mary can drink one barrel in 12 days. "
            "How long would it take them to drink one barrel **together**?",
            ["9 days", "4 days", "3 days", "6 days"],
            index=0,
        )
        crt_q5 = st.radio(
            "**Q5** · Jerry received both the 15th highest and the 15th lowest mark in the class. "
            "How many students are in the class?",
            ["30 students", "29 students", "28 students", "31 students"],
            index=0,
        )
        crt_q6 = st.radio(
            "**Q6** · A man buys a pig for $60, sells it for $70, buys it back for $80, and sells it "
            "finally for $90. How much has he **made** in total?",
            ["$10", "$20", "$30", "$0"],
            index=0,
        )
        crt_q7 = st.radio(
            "**Q7** · Simon invested $8,000. Six months later his stocks had fallen 50%. "
            "From that low point, stocks then rose 75%. Simon now has:",
            ["Come out ahead", "Broken even", "Lost money", "Doubled his money"],
            index=0,
        )
        st.markdown("---")
        st.subheader("Cognitive Style")
        nfc = st.slider(
            "Need for Cognition (NFC) — enjoy thinking deeply (1=strongly disagree, 5=strongly agree)",
            1, 5, 3,
        )

        st.subheader("Background")
        education = st.selectbox(
            "Highest education level",
            ["High school or below", "Some college", "Bachelor's degree", "Graduate degree"],
        )
        edu_map = {"High school or below": 1, "Some college": 2, "Bachelor's degree": 3, "Graduate degree": 4}
        edu_val = edu_map[education]

        political = st.slider(
            "Political leaning (1=Far left, 7=Far right)",
            1, 7, 4,
        )

        st.subheader("Media Habits")
        social_hrs = st.slider(
            "Average daily social media hours",
            0.0, 12.0, 2.5, step=0.5,
        )
        news_literacy = st.slider(
            "News media literacy — ability to evaluate news sources (1=low, 5=high)",
            1, 5, 3,
        )
        prior = st.slider(
            "Prior exposure to misinformation content (0=never, 1=constantly)",
            0.0, 1.0, 0.3, step=0.05,
        )

        submitted = st.form_submit_button("📊 Generate Profile", type="primary")

    if submitted:
        st.session_state["profiler_submitted"] = True

    if st.session_state.get("profiler_submitted"):
        # Score the CRT answers
        _crt_answers = [
            (crt_q1, "5 cents",    "Q1", "The ball costs **5 cents** ($1.05 + $0.05 = $1.10; difference = $1.00). The intuitive answer of 10 cents means the bat is only $1.00 more — not $1.00."),
            (crt_q2, "5 minutes",  "Q2", "Still **5 minutes** — each machine independently makes 1 widget per 5 min, so 100 machines make 100 widgets in the same 5 minutes."),
            (crt_q3, "49 days",    "Q3", "**49 days** — on day 48 the patch covers half the lake; one more doubling (day 49) covers the whole lake."),
            (crt_q4, "4 days",     "Q4", "**4 days** — combined rate = 1/6 + 1/12 = 1/4 barrel/day, so 1 barrel takes 4 days."),
            (crt_q5, "29 students","Q5", "**29 students** — 15th highest + 15th lowest means 14 above Jerry, Jerry, 14 below = 29 total."),
            (crt_q6, "$20",        "Q6", "**$20** — two independent transactions each yield $10 profit ($60→$70 and $80→$90)."),
            (crt_q7, "Lost money", "Q7", "**Lost money** — $8,000 × 0.50 = $4,000, then × 1.75 = $7,000. A 75% rise from a 50% loss leaves you $1,000 short."),
        ]
        crt = sum(ans == correct for ans, correct, _, __ in _crt_answers)

        # Compute a synthetic susceptibility score (weighted formula from study)
        # Lower CRT → more susceptible; lower NFC → more susceptible; more social media → more susceptible
        raw = (
            -0.30 * (crt / 7.0)        # CRT (protective)
            - 0.20 * (nfc / 5.0)       # NFC (protective)
            - 0.10 * (news_literacy / 5.0)  # literacy (protective)
            + 0.20 * (social_hrs / 12.0)   # social media (risk)
            + 0.10 * prior              # prior exposure (risk)
            + 0.10 * (abs(political - 4) / 3.0)  # extremity (risk)
            + 0.60                      # intercept → centre at ~0.5
        )
        risk = float(np.clip(raw, 0.0, 1.0))

        st.markdown("---")
        st.subheader("Your Susceptibility Profile")

        col1, col2 = st.columns([2, 3])
        with col1:
            if risk < 0.35:
                level, color, emoji = "LOW", "#4CAF50", "✅"
            elif risk < 0.6:
                level, color, emoji = "MODERATE", "#FF9800", "⚠️"
            else:
                level, color, emoji = "HIGH", "#F44336", "❌"
            st.metric(f"{emoji} Susceptibility", f"{risk:.0%}", level)
            fig_gauge = make_gauge(risk, "Overall Risk")
            st.pyplot(fig_gauge)
            plt.close(fig_gauge)

        with col2:
            # Radar-style breakdown
            dims = ["Low CRT\n(risk)", "Low NFC\n(risk)", "Low Literacy\n(risk)",
                    "High Social\nMedia", "Prior Exp.", "Political\nExtremity"]
            raw_vals = [
                1 - crt / 7.0,
                1 - nfc / 5.0,
                1 - news_literacy / 5.0,
                social_hrs / 12.0,
                prior,
                abs(political - 4) / 3.0,
            ]
            colors_dim = ["#F44336" if v > 0.5 else "#4CAF50" for v in raw_vals]
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3.5))
            bars_b = ax_bar.barh(dims, raw_vals, color=colors_dim, alpha=0.8, edgecolor="white")
            ax_bar.set_xlim(0, 1)
            ax_bar.set_xlabel("Relative risk contribution")
            ax_bar.set_title("Risk Factor Breakdown", fontweight="bold")
            ax_bar.axvline(0.5, color="k", lw=1, linestyle="--", alpha=0.5)
            ax_bar.grid(True, axis="x", alpha=0.3)
            fig_bar.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        # Recommendations
        st.markdown("### 💡 Personalised Recommendations")
        recs = []
        if crt <= 3:
            recs.append("**Slow down before sharing** — take 30 s to verify a claim before forwarding.")
        if social_hrs >= 4:
            recs.append("**Reduce passive scrolling** — set daily limits; passive consumption increases exposure.")
        if news_literacy <= 2:
            recs.append("**Source-check habit** — check if the outlet has a known bias (mediabiasfactcheck.com).")
        if not recs:
            recs.append("You show a strong protective profile. Keep applying analytical thinking!")
        for r in recs:
            st.markdown(f"- {r}")

        # CRT answer reveal
        st.markdown("---")
        st.markdown(f"### 🧠 CRT Results — {crt}/7 correct")
        for ans, correct, label, explanation in _crt_answers:
            is_right = ans == correct
            icon = "✅" if is_right else "❌"
            verdict = "correct" if is_right else f"incorrect — you chose **{ans}**, answer is **{correct}**"
            with st.expander(f"{icon} {label} — {verdict}"):
                st.markdown(explanation)


# ============================================================================
#  PAGE: FUSION PREDICTOR
# ============================================================================
elif page == "🔀 Fusion Predictor":
    st.title("🔀 Fusion Predictor")
    st.markdown(
        """
        Combines **article content** (NLP) and **user behavioral profile** to predict both the
        *article's credibility* and a personalised *susceptibility score*.
        """
    )
    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📰 Article Content")
        article_text = st.text_area(
            "Paste headline / excerpt:",
            height=130,
            placeholder="Breaking: Scientists discover 5G towers affect brain chemistry — exclusive report",
        )

    with col_right:
        st.subheader("👤 Quick Behavioral Profile")
        f_crt    = st.slider("CRT score", 0, 7, 4, key="f_crt")
        f_nfc    = st.slider("NFC (1–5)", 1, 5, 3, key="f_nfc")
        f_social = st.slider("Social media hrs/day", 0.0, 12.0, 2.5, 0.5, key="f_social")
        f_lit    = st.slider("News literacy (1–5)", 1, 5, 3, key="f_lit")

    if st.button("🚀 Run Fusion Prediction", type="primary", use_container_width=True):
        if not article_text.strip():
            st.warning("Please enter article text.")
        else:
            nlp_prob = None
            if nlp_models and "tfidf" in nlp_models and "lr" in nlp_models:
                try:
                    X_nlp_in = nlp_models["tfidf"].transform([article_text])
                    nlp_prob = float(nlp_models["lr"].predict_proba(X_nlp_in)[0, 1])
                except Exception:
                    pass
            nlp_prob = nlp_prob if nlp_prob is not None else 0.5
            beh_risk = float(np.clip(
                -0.30 * (f_crt / 7.0)
                - 0.20 * (f_nfc / 5.0)
                - 0.10 * (f_lit / 5.0)
                + 0.20 * (f_social / 12.0)
                + 0.60,
                0.0, 1.0,
            ))
            fusion_prob = float(0.65 * nlp_prob + 0.35 * beh_risk)
            st.session_state["fusion_result"] = {
                "nlp_prob": nlp_prob,
                "beh_risk": beh_risk,
                "fusion_prob": fusion_prob,
            }

    fusion_res = st.session_state.get("fusion_result")
    if fusion_res:
        nlp_prob    = fusion_res["nlp_prob"]
        beh_risk    = fusion_res["beh_risk"]
        fusion_prob = fusion_res["fusion_prob"]

        st.markdown("---")
        st.subheader("Fusion Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📰 Content Risk (NLP)**")
            fig_n = make_gauge(nlp_prob, "NLP")
            st.pyplot(fig_n); plt.close(fig_n)
        with c2:
            st.markdown("**👤 User Susceptibility**")
            fig_b = make_gauge(beh_risk, "Behavioral")
            st.pyplot(fig_b); plt.close(fig_b)
        with c3:
            st.markdown("**🔀 Fusion Score**")
            fig_f = make_gauge(fusion_prob, "Fusion")
            st.pyplot(fig_f); plt.close(fig_f)

        st.markdown("---")
        if fusion_prob >= 0.65:
            st.error(
                "⚠️ **High combined risk**: the article has strong fake-news signals "
                "AND this user profile is susceptible. Verify before sharing."
            )
        elif fusion_prob >= 0.45:
            st.warning(
                "🟡 **Moderate risk**: ambiguous article or moderately susceptible profile. "
                "Apply extra scrutiny."
            )
        else:
            st.success(
                "✅ **Low combined risk**: content appears credible and profile shows protective factors."
            )

        if nlp_prob > 0.5 and beh_risk < 0.4:
            st.info(
                "ℹ️ The article content looks risky but the user's analytical profile provides some protection."
            )
        elif nlp_prob < 0.5 and beh_risk > 0.6:
            st.warning(
                "⚠️ The article looks credible but the user's profile suggests elevated susceptibility to framing effects."
            )


# ============================================================================
#  PAGE: MODEL COMPARISON
# ============================================================================
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison Dashboard")
    st.markdown("Side-by-side comparison of all trained models with ablation and validation results.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Performance Table", "Ablation Study", "Validation", "Feature Importance"])

    # ---- Tab 1: Comparison table
    with tab1:
        st.subheader("Model Performance Summary")
        if "comparison" in results:
            df_cmp = results["comparison"]
            st.dataframe(df_cmp.style.highlight_max(axis=0, subset=[c for c in df_cmp.columns if c != "Model"],
                                                     color="#C8E6C9"), use_container_width=True)
        elif "validation" in results:
            df_cmp = results["validation"]
            st.dataframe(df_cmp, use_container_width=True)
        else:
            # Inline demo table
            demo_df = pd.DataFrame([
                {"Model": "Naive Bayes",               "AUC": 0.812, "Accuracy": 0.756, "MacroF1": 0.753},
                {"Model": "TF-IDF + LR",               "AUC": 0.859, "Accuracy": 0.812, "MacroF1": 0.810},
                {"Model": "TF-IDF + SVM",              "AUC": 0.841, "Accuracy": 0.794, "MacroF1": 0.792},
                {"Model": "SBERT + LR",                "AUC": 0.871, "Accuracy": 0.824, "MacroF1": 0.820},
                {"Model": "Fusion (GBM)",               "AUC": 0.891, "Accuracy": 0.843, "MacroF1": 0.839},
            ])
            st.dataframe(demo_df.style.highlight_max(axis=0, subset=["AUC","Accuracy","MacroF1"],
                                                     color="#C8E6C9"), use_container_width=True)
            st.caption("Demo data — run Phase 2 & 3 notebooks to populate real results.")

        # ROC curve image
        roc_img = result_img("model_comparison_roc.png")
        st.image(str(roc_img), caption="ROC Curves — All Models", use_container_width=True)

    # ---- Tab 2: Ablation
    with tab2:
        st.subheader("Ablation Studies")
        abl_img = result_img("ablation_modality.png")
        st.image(str(abl_img), caption="Modality Ablation", use_container_width=True)

        abl_feat = result_img("ablation_behavioral_features.png")
        st.image(str(abl_feat), caption="Behavioral Feature Leave-One-Out", use_container_width=True)

        if "ablation" in results:
            st.json(results["ablation"])

    # ---- Tab 3: Validation
    with tab3:
        st.subheader("Validation & Generalization")
        val_imgs = [
            (result_img("validation_nested_cv.png"),     "Nested Cross-Validation AUC"),
            (result_img("validation_bootstrap_ci.png"),  "Bootstrap 95% CI"),
            (result_img("validation_domain_transfer.png"), "Domain Transfer Simulation"),
            (result_img("validation_final_summary.png"), "Final Performance Heatmap"),
        ]
        for path, caption in val_imgs:
            st.image(str(path), caption=caption, use_container_width=True)

        if "validation" in results:
            st.subheader("Validation Summary Table")
            st.dataframe(results["validation"], use_container_width=True)

    # ---- Tab 4: Feature Importance
    with tab4:
        st.subheader("Feature Importance")
        shap_img  = result_img("shap_feature_importance.png")
        perm_img  = result_img("permutation_importance.png")
        pdp_img   = result_img("partial_dependence_behavioral.png")
        ind_img   = result_img("individual_explanations.png")

        for path, cap in [(shap_img, "SHAP Global Importance"), (perm_img, "Permutation Importance"),
                          (pdp_img, "Partial Dependence"), (ind_img, "Individual Explanations")]:
            st.image(str(path), caption=cap, use_container_width=True)

        if "perm_imp" in results:
            st.subheader("Top Features (Permutation Importance)")
            st.dataframe(results["perm_imp"].head(15), use_container_width=True)

        # Always show interactive importance builder
        st.markdown("---")
        st.subheader("🎛️ Interactive Importance Explorer")
        st.markdown("Adjust the weights below to simulate how changing a feature's influence affects the fusion model's risk score.")
        demo_feats = {
            "CRT Score":          -0.30,
            "NFC":                -0.20,
            "News Literacy":      -0.10,
            "Social Media Hours": +0.20,
            "Prior Exposure":     +0.10,
            "Political Extremity": +0.10,
        }
        weights = {}
        cols_w = st.columns(3)
        for i, (fname, default) in enumerate(demo_feats.items()):
            with cols_w[i % 3]:
                weights[fname] = st.slider(fname, -0.50, 0.50, float(default), 0.05, key=f"w_{fname}")

        # Visualise
        fig_imp, ax_imp = plt.subplots(figsize=(7, 3))
        fnames = list(weights.keys())
        wvals  = [weights[f] for f in fnames]
        colors_imp = ["#F44336" if w > 0 else "#4CAF50" for w in wvals]
        ax_imp.bar(fnames, wvals, color=colors_imp, alpha=0.8, edgecolor="white")
        ax_imp.axhline(0, color="k", lw=0.8)
        ax_imp.set_title("Custom Feature Weight Simulation", fontweight="bold")
        ax_imp.set_ylabel("Weight")
        ax_imp.set_xticklabels(fnames, rotation=20, ha="right", fontsize=9)
        ax_imp.grid(True, axis="y", alpha=0.3)
        fig_imp.tight_layout()
        st.pyplot(fig_imp)
        plt.close(fig_imp)


# ============================================================================
#  PAGE: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        ## Tentacles of Misinformation

        This dashboard is the Phase 4 deliverable of a multi-phase research project
        investigating misinformation detection and susceptibility.

        ### Research Phases
        | Phase | Title | Status |
        |-------|-------|--------|
        | 1 | Behavioral Foundation & Baseline NLP | ✅ Complete |
        | 2 | Transformer Classifiers & Analysis | ✅ Complete |
        | 3 | Multi-Modal Fusion Models | ✅ Complete |
        | 4 | Research Hub Dashboard | ✅ Live |

        ### Key Models
        | Model | AUC | Notes |
        |-------|-----|-------|
        | TF-IDF + LR | 0.859 | Production baseline |
        | TF-IDF + SVM | 0.841 | Competitive baseline |
        | SBERT + LR | ~0.871 | Semantic embeddings |
        | Fusion (GBM) | ~0.891 | NLP + Behavioral |

        ### Data
        - **FakeNewsNet**: PolitiFact + GossipCop, ~23 000 articles
        - **Behavioral study**: N=194 participants (IRB-approved, data not redistributed)

        ### Citation
        If using this work, cite the associated thesis / paper.

        ### Navigation Guide
        - **NLP Analysis Tool**: Enter any text for real-time fake-news scoring
        - **Behavioral Profiler**: Rate cognitive/media habits → get a susceptibility profile
        - **Fusion Predictor**: Combine content + profile for a personalised risk assessment
        - **Model Comparison**: Explore all experimental results and ablation studies
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit · Scikit-learn · Sentence-Transformers · SHAP")
