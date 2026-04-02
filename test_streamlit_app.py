import sys
from pathlib import Path

# Test model loading
sys.path.insert(0, str(Path.cwd()))

from dashboards.streamlit.app import load_model

print('Testing model loading...')
vectorizer, model = load_model()

if vectorizer is not None and model is not None:
    print('✅ Models loaded successfully!')
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
else:
    print('❌ Model loading failed')
