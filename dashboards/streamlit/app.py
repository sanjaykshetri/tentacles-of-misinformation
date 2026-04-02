import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import numpy as np

# Configuration
st.set_page_config(
    page_title="Misinformation Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path resolution for HF Space (models/ in same directory as app.py)
MODEL_DIR = Path(__file__).parent / "models"

@st.cache_resource
def load_model():
    """Load baseline TF-IDF + Logistic Regression model."""
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    model_path = MODEL_DIR / "logistic_regression_baseline.joblib"
    
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model

def predict_fake_news(text, vectorizer, model):
    """Make prediction with confidence scores."""
    try:
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        confidence = model.predict_proba(text_vector)[0]
        return prediction, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Page configuration
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select page:", ["🏠 Classifier", "📊 Model Info", "ℹ️ About"])

# Load models with error display
try:
    vectorizer, model = load_model()
    if vectorizer is None or model is None:
        st.error("❌ Models failed to load (None returned)")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

# ============================================================================
# PAGE 1: Interactive Classifier
# ============================================================================
if page == "🏠 Classifier":
    st.title("🔍 Misinformation Classifier")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text to Classify")
        user_text = st.text_area(
            "Paste article text, social media post, or headline:",
            height=200,
            placeholder="Example: 'Study shows eating chocolate prevents heart disease...'"
        )
    
    with col2:
        st.info(
            "**Model**: Logistic Regression\n"
            "**Features**: TF-IDF (Top 5K terms)\n"
            "**Accuracy**: 81.2%\n"
            "**Data**: FakeNewsNet (21.7K articles)",
            icon="ℹ️"
        )
    
    st.markdown("---")
    
    if st.button("🎯 Classify Text", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to classify")
        elif vectorizer is None or model is None:
            st.error("❌ Model failed to load. Please refresh the page.")
        else:
            with st.spinner("Analyzing text..."):
                prediction, confidence = predict_fake_news(user_text, vectorizer, model)
                
                if prediction is not None:
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 0:
                            st.success("## ✅ Likely REAL", icon="✅")
                            confidence_real = confidence[0]
                            st.metric("Confidence (Real)", f"{confidence_real:.1%}")
                            st.progress(confidence_real)
                        else:
                            st.error("## ❌ Likely FAKE", icon="⚠️")
                            confidence_fake = confidence[1]
                            st.metric("Confidence (Fake)", f"{confidence_fake:.1%}")
                            st.progress(confidence_fake)
                    
                    with col2:
                        st.markdown("### Prediction Details")
                        df_conf = pd.DataFrame({
                            "Class": ["Real", "Fake"],
                            "Probability": [f"{confidence[0]:.1%}", f"{confidence[1]:.1%}"]
                        })
                        st.dataframe(df_conf, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Key terms extracted (top TF-IDF terms)
                    st.markdown("### 🔑 Key Terms Detected")
                    try:
                        feature_names = vectorizer.get_feature_names_out()
                        text_vector = vectorizer.transform([user_text])
                        
                        # Get top terms by TF-IDF score
                        tfidf_scores = text_vector.toarray()[0]
                        top_indices = np.argsort(tfidf_scores)[-10:][::-1]
                        
                        top_terms = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
                        
                        if top_terms:
                            terms_col1, terms_col2 = st.columns(2)
                            for idx, (term, score) in enumerate(top_terms):
                                if idx < len(top_terms) // 2:
                                    with terms_col1:
                                        st.caption(f"📌 {term}: {score:.3f}")
                                else:
                                    with terms_col2:
                                        st.caption(f"📌 {term}: {score:.3f}")
                    except Exception as e:
                        st.info("Unable to extract key terms")

# ============================================================================
# PAGE 2: Model Information
# ============================================================================
elif page == "📊 Model Info":
    st.title("📊 Model Performance & Details")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Architecture")
        st.markdown("""
        **Type**: Supervised Classification  
        **Algorithm**: Logistic Regression  
        **Vectorization**: TF-IDF  
        **Vocabulary Size**: 5,000 terms  
        **Regularization**: L2 (C=1.0)
        """)
    
    with col2:
        st.subheader("📈 Performance Metrics")
        metrics_data = {
            "Metric": ["Accuracy", "Precision (Fake)", "Recall (Fake)", "F1-Score"],
            "Score": ["81.2%", "78.5%", "74.2%", "0.763"]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📚 Training Data")
    st.markdown("""
    - **Dataset**: FakeNewsNet (GossipCop + PolitiFact)
    - **Total Articles**: 21,754
    - **Fake News**: 11,071 (50.9%)
    - **Real News**: 10,683 (49.1%)
    - **Time Period**: 2016-2019
    - **Sources**: News websites, social media articles
    """)
    
    st.markdown("---")
    
    st.subheader("⚙️ Feature Engineering Pipeline")
    st.markdown("""
    1. **Text Preprocessing**
       - Lowercase conversion
       - Removal of special characters
       - Tokenization
    
    2. **Feature Extraction**
       - TF-IDF Vectorization
       - Unigrams and bigrams
       - Minimum document frequency: 2
       - Maximum document frequency: 0.8
    
    3. **Model Training**
       - Train/test split: 80/20
       - Stratified sampling (maintain class balance)
       - Hyperparameter tuning via cross-validation
    """)

# ============================================================================
# PAGE 3: About
# ============================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    **Tentacles of Misinformation** is a comprehensive system for detecting and modeling 
    fake news and misinformation at scale. This classifier is one component of a larger 
    platform that combines:
    
    - **Behavioral Analysis**: Understanding why people believe misinformation
    - **Detection Systems**: ML models for fake news identification
    - **Spread Modeling**: SEIR epidemic models for misinformation propagation
    - **Fusion Architectures**: Multi-modal approaches combining text + network features
    - **Production Infrastructure**: Deployment-ready systems architecture
    
    ---
    
    ## 🚀 Technology Stack
    
    - **ML Framework**: scikit-learn
    - **Interface**: Streamlit
    - **Vectorization**: TF-IDF
    - **Classification**: Logistic Regression (baseline) + SVM + Transformers (advanced)
    - **Data**: FakeNewsNet, behavioral survey data
    
    ---
    
    ## 📖 Learn More
    
    - **📚 [Full Book/Documentation](https://sanjaykshetri.github.io/tentacles-of-misinformation/)**
    - **💻 [Source Code](https://github.com/sanjaykshetri/tentacles-of-misinformation)**
    - **🔬 [Research Details](https://github.com/sanjaykshetri/tentacles-of-misinformation/blob/main/docs/TECHNICAL_ROADMAP.md)**
    
    ---
    
    ## 📧 Contact
    
    Built by Sanjay Shetri · [GitHub](https://github.com/sanjaykshetri/) 
    
    Questions? Feedback? Found a bug? Open an issue on GitHub or reach out directly.
    """)

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.caption("🔗 [GitHub](https://github.com/sanjaykshetri/tentacles-of-misinformation)")

with col_footer2:
    st.caption("📚 [Documentation](https://sanjaykshetri.github.io/tentacles-of-misinformation/)")

with col_footer3:
    st.caption("⭐ Built with Streamlit")
