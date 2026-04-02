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

# Path resolution - works from any working directory
MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# Fallback URLs for models (in case not in local repo)
MODEL_URLS = {
    "vectorizer": "https://github.com/sanjaykshetri/tentacles-of-misinformation/raw/main/models/tfidf_vectorizer.joblib",
    "model": "https://github.com/sanjaykshetri/tentacles-of-misinformation/raw/main/models/logistic_regression_baseline.joblib"
}

@st.cache_resource
def load_model():
    """Load baseline TF-IDF + Logistic Regression model with error handling."""
    import urllib.request
    import tempfile
    
    try:
        # First, try to load from local directory
        if (MODEL_DIR / "tfidf_vectorizer.joblib").exists() and (MODEL_DIR / "logistic_regression_baseline.joblib").exists():
            vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
            model = joblib.load(MODEL_DIR / "logistic_regression_baseline.joblib")
            return vectorizer, model
    except Exception as e:
        st.warning(f"⚠️ Local models not found, downloading from GitHub...")
    
    # Fallback: download from GitHub
    try:
        with st.spinner("📥 Downloading models from GitHub (first time only)..."):
            temp_dir = tempfile.gettempdir()
            
            # Download vectorizer
            vectorizer_file = Path(temp_dir) / "tfidf_vectorizer.joblib"
            if not vectorizer_file.exists():
                urllib.request.urlretrieve(MODEL_URLS["vectorizer"], vectorizer_file)
            
            # Download model
            model_file = Path(temp_dir) / "logistic_regression_baseline.joblib"
            if not model_file.exists():
                urllib.request.urlretrieve(MODEL_URLS["model"], model_file)
            
            vectorizer = joblib.load(vectorizer_file)
            model = joblib.load(model_file)
            return vectorizer, model
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.info("Please ensure model files are available or check your internet connection")
        return None, None

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

vectorizer, model = load_model()

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
