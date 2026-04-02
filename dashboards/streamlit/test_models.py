import sys
sys.path.insert(0, '.')
from pathlib import Path
import joblib

MODEL_DIR = Path('../..') / 'models'
print(f'Looking for models in: {MODEL_DIR}')
print(f'Exists: {MODEL_DIR.exists()}')

if (MODEL_DIR / 'tfidf_vectorizer.joblib').exists():
    print('✅ TF-IDF vectorizer found')
    v = joblib.load(MODEL_DIR / 'tfidf_vectorizer.joblib')
    print(f'   Loaded: {type(v).__name__} with vocab size {len(v.get_feature_names_out())}')
else:
    print('❌ Vectorizer not found')

if (MODEL_DIR / 'logistic_regression_baseline.joblib').exists():
    print('✅ Model found')
    m = joblib.load(MODEL_DIR / 'logistic_regression_baseline.joblib')
    print(f'   Loaded: {type(m).__name__}')
    
    # Test prediction
    test_vec = v.transform(['this is fake news'])
    pred = m.predict(test_vec)[0]
    conf = m.predict_proba(test_vec)[0]
    print(f'   Test prediction: {"FAKE" if pred else "REAL"} ({conf.max():.1%})')
else:
    print('❌ Model not found')
