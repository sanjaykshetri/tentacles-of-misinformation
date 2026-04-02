# Streamlit Misinformation Classifier

A simple, production-ready interface for the TF-IDF + Logistic Regression baseline misinformation detector.

## Features

- **Interactive Text Classification**: Paste text and get fake/real predictions with confidence scores
- **Model Information**: View detailed performance metrics and training data info
- **Key Term Extraction**: See which terms influenced the prediction
- **Real Model**: Uses trained baseline (81.2% accuracy on FakeNewsNet)

## Local Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Deployment

### Hugging Face Spaces

The app is deployed to Hugging Face Spaces:
[🚀 Open Live Demo](https://huggingface.co/spaces/sanjaykshetri/tentacles)

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

## Model Details

- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF (5K vocabulary)
- **Accuracy**: 81.2%
- **Data**: FakeNewsNet (21.7K articles, 2016-2019)
- **Training Set**: 80/20 split, stratified

## Project Links

- 📚 [Full Documentation](https://sanjaykshetri.github.io/tentacles-of-misinformation/)
- 💻 [GitHub Repository](https://github.com/sanjaykshetri/tentacles-of-misinformation)
- 🔬 [Research Background](https://github.com/sanjaykshetri/tentacles-of-misinformation/blob/main/docs/TECHNICAL_ROADMAP.md)

## License

MIT License - See main repository for details
