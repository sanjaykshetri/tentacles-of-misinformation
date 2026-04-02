# Streamlit App Deployment Guide

This guide walks you through deploying the misinformation classifier to **Hugging Face Spaces** (free, shareable, no setup).

## Option 1: Deploy to Hugging Face Spaces (Recommended - Free & Easy)

### Step 1: Create HF Account & Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `misinformation-classifier` (or your choice)
   - **License**: `Open License` 
   - **Space SDK**: Select **Streamlit**
4. Click "Create Space"

### Step 2: Connect Your Repository

```bash
# Clone the HF Space repo (you'll get the URL after creating it)
git clone https://huggingface.co/spaces/YOUR_USERNAME/misinformation-classifier
cd misinformation-classifier

# Copy files from this repo
cp -r ../tentacles-of-misinformation/dashboards/streamlit/* .

# Ensure models directory exists relative to the app
mkdir -p ../../models
cp ../tentacles-of-misinformation/models/*.joblib ../../models/
```

**OR** (simpler - if you have direct GitHub access):

1. Go to your Space's "Files" tab
2. Click "Clone repository"
3. Enter: `https://github.com/sanjaykshetri/tentacles-of-misinformation/dashboards/streamlit`

### Step 3: Add Model Files

HuggingFace Spaces has a 50GB storage limit. The models are small (~30MB total):

**Option A: Upload via Web UI**
- Go to Files tab
- Upload the 3 `.joblib` files from `models/` folder:
  - `tfidf_vectorizer.joblib`
  - `logistic_regression_baseline.joblib`
  - `linear_svm_baseline.joblib`

**Option B: Git LFS (if models are >100MB)**
- Install [Git LFS](https://git-lfs.github.com/)
- Run in your cloned repo:
```bash
git lfs install
git lfs track "*.joblib"
git add .
git commit -m "Add models with Git LFS"
git push
```

### Step 4: Push & Deploy

```bash
# The app auto-deploys when you push
git add .
git commit -m "Deploy Streamlit classifier"
git push
```

The app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/misinformation-classifier`

---

## Option 2: Deploy to Railway (5-min setup, then $5/month)

### Step 1: Create Account
- Go to [railway.app](https://railway.app)
- Sign up with GitHub

### Step 2: Create New Project
- Click "Create New"
- Select "GitHub Repo"
- Choose: `tentacles-of-misinformation`

### Step 3: Add Streamlit Config
- In your project, create environment variables in Railway dashboard:
  - `STREAMLIT_SERVER_HEADLESS=true`
  - `STREAMLIT_SERVER_PORT=8501`

### Step 4: Deploy
- Railway auto-deploys on every GitHub push
- Your app will be at: `https://your-project.up.railway.app`

---

## Option 3: Deploy to Heroku (Free tier removed, now $7+/month)

Heroku deprecated free tier in Nov 2022. Still possible but requires payment. Skip unless you prefer it.

---

## Option 4: Docker-Based Deployment

If you want full control:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY dashboards/streamlit /app
COPY models /models

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]
```

Build & run:
```bash
docker build -t misinformation-classifier .
docker run -p 8501:8501 misinformation-classifier
```

---

## 📊 Recommended: Hugging Face Spaces

**Why?**
- ✅ Free forever
- ✅ No credit card needed
- ✅ Auto deployment from GitHub
- ✅ Easy model file uploads
- ✅ Built for ML demos
- ✅ Shareable public links

---

## Troubleshooting

### "Models not found" error
**Fix**: Ensure model files are in the same directory structure. The app looks for:
```
├── app.py
├── requirements.txt
└── ../../models/
    ├── tfidf_vectorizer.joblib
    ├── logistic_regression_baseline.joblib
    └── linear_svm_baseline.joblib
```

### "Module not found" error
**Fix**: Run `pip install -r requirements.txt`

### App loads but predictions fail
**Fix**: Check that joblib can actually load the models by running:
```bash
python -c "
import joblib
from pathlib import Path
MODEL_DIR = Path('../../models')
v = joblib.load(MODEL_DIR / 'tfidf_vectorizer.joblib')
m = joblib.load(MODEL_DIR / 'logistic_regression_baseline.joblib')
print(f'✅ Loaded. Vocab size: {len(v.get_feature_names_out())}')
"
```

---

## Next Steps After Deployment

1. **Share the link** in your portfolio/resume
2. **Add to README**: Update the live demo link
3. **Test thoroughly**: Try edge cases
4. **Monitor logs**: Check Space/Railway logs for errors

---

## Key Files

- `app.py` - Main Streamlit app
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `README.md` - Local usage guide

For questions, see the main [GitHub repo](https://github.com/sanjaykshetri/tentacles-of-misinformation).
