# Hugging Face Space Deployment - Quick Setup

This guide covers the **complete deployment to Hugging Face Spaces** (5-10 minutes).

## Prerequisites

- Hugging Face account (free at https://huggingface.co)
- Git installed
- (Optional) git-xet for large file handling

## Step 1: Create a Hugging Face Space

1. **Log in** to https://huggingface.co
2. Go to **Spaces** (top menu)
3. Click **Create new Space**
4. Fill in:
   - **Space name**: `misinformation-classifier` (or similar)
   - **License**: `Open License`
   - **Space SDK**: Select **Streamlit**
   - **Visibility**: `Public` (for portfolio)
5. Click **Create Space**

✅ Your space is now created. You'll see a repository URL like:
```
https://huggingface.co/spaces/YOUR_USERNAME/misinformation-classifier
```

## Step 2: Clone Your Space Repository

```bash
# Navigate to where you want to work
cd C:\Users\sanja\OneDrive\Documents

# Clone your new space
git clone https://huggingface.co/spaces/YOUR_USERNAME/misinformation-classifier
cd misinformation-classifier

# Add HF remote
git remote rename origin huggingface
```

## Step 3: Copy App Files

From the main repo `dashboards/streamlit/`:

```bash
# Copy app files to space directory
cp ..\..\GitHub\tentacles-of-misinformation\dashboards\streamlit\app.py .
cp ..\..\GitHub\tentacles-of-misinformation\dashboards\streamlit\requirements.txt .
cp -r ..\..\GitHub\tentacles-of-misinformation\dashboards\streamlit\.streamlit .

# Copy model files
mkdir -p models
cp ..\..\GitHub\tentacles-of-misinformation\models\*.joblib models/
```

## Step 4: Adjust Paths for HF Space

Edit `app.py` to work with HF Space structure. Change this line:

```python
# FROM:
MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# TO:
MODEL_DIR = Path(__file__).parent / "models"
```

## Step 5: Push to Hugging Face

```bash
git add .
git commit -m "Add misinformation classifier app"
git push huggingface main

# (If main branch doesn't exist, use: git push huggingface -u main)
```

✅ **Your app is now live!** Access it at:
```
https://huggingface.co/spaces/YOUR_USERNAME/misinformation-classifier
```

## Updating Your App

Any time you update the app:

```bash
git add .
git commit -m "Update: description of changes"
git push huggingface main
```

The space **auto-rebuilds** within 1-2 minutes.

## Troubleshooting

### "Models not found" error
- Confirm `models/` directory exists in space root
- Verify all three `.joblib` files are present:
  - `tfidf_vectorizer.joblib`
  - `logistic_regression_baseline.joblib`
  - `linear_svm_baseline.joblib`

### "ModuleNotFoundError"
- Check `requirements.txt` has all dependencies
- Clear HF Space cache (Space settings → Delete space app)

### Large file upload limits
Use git-xet for files >100MB:
```bash
# Install git-xet
winget install git-xet

# Then use normal git commands — git-xet handles large files automatically
git push huggingface main
```

## Links

- **Live Demo**: https://huggingface.co/spaces/sanjaykshetri/tentacles
- **Main Repo**: https://github.com/sanjaykshetri/tentacles-of-misinformation
- **Documentation**: https://sanjaykshetri.github.io/tentacles-of-misinformation/

## After Deployment

1. Add the live link to your README:
   ```
   🎮 **[Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/misinformation-classifier)**
   ```

2. Update portfolio/resume with link

3. Test with various text samples to verify predictions

---

**Total time**: ~5-10 minutes  
**Cost**: Free (HF Spaces are free)  
**Result**: Live, shareable classifier anyone can use
