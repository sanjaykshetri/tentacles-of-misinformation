import joblib
from pathlib import Path

# Direct model loading test (without Streamlit decorators)
MODEL_DIR = Path(__file__).parent / "models"

print("Testing model loading (direct joblib)...")
print(f"Looking for models in: {MODEL_DIR}")
print(f"Directory exists: {MODEL_DIR.exists()}")

if MODEL_DIR.exists():
    print(f"Files in models/: {list(MODEL_DIR.glob('*.joblib'))}")

try:
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "logistic_regression_baseline.joblib")
    
    print('\n✅ Models loaded successfully!')
    print(f'   Vectorizer: {type(vectorizer).__name__}')
    print(f'   Model: {type(model).__name__}')
    print(f'   Vocabulary size: {len(vectorizer.get_feature_names_out())}')
    
    # Test a prediction
    test_text = 'Breaking: Study shows chocolate cures cancer'
    text_vec = vectorizer.transform([test_text])
    pred = model.predict(text_vec)[0]
    conf = model.predict_proba(text_vec)[0]
    
    print(f'\n   Test prediction:')
    print(f'      Text: "{test_text}"')
    print(f'      Predicted: {"FAKE" if pred == 1 else "REAL"}')
    print(f'      Confidence: {conf.max():.1%}')
    
except FileNotFoundError as e:
    print(f'❌ Model files not found: {e}')
    print(f"\nSearching for joblib files in: {MODEL_DIR}")
    if MODEL_DIR.exists():
        files = list(MODEL_DIR.glob('*.joblib'))
        print(f"Found: {files}")
except Exception as e:
    print(f'❌ Error: {e}')
