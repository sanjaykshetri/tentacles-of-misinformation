# Dashboards

## Overview

Interactive dashboards for exploring misinformation susceptibility insights, model predictions, and policy recommendations.

## Components

- **streamlit/** - Interactive Python dashboards for research exploration
- **powerbi/** - Business intelligence dashboards for stakeholders

## Streamlit Apps

### 1. Behavioral Explorer
- Visualize distributions of cognitive features
- Interactive demographic breakdowns
- Statistical summary cards

### 2. NLP Model Dashboard
- Input text to classify misinformation
- View classification confidence scores
- Explore similar texts in corpus

### 3. Fusion Model Risk Predictor
- Enter behavioral profile
- Analyze text content
- Get personalized susceptibility risk assessment
- Explainability via feature attribution

### 4. Research Hub
- Dataset summaries
- Model performance metrics
- Publication-ready visualizations

## Technologies

- **Streamlit** - Python web app framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation and display

## Deployment

- Local: `streamlit run app.py`
- Cloud: Streamlit Cloud integration
- Power BI: Direct data connections

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
cd dashboards/streamlit
streamlit run app.py
```

## Data Sources

- Behavioral analysis results
- NLP model predictions
- Fusion model outputs

## Next Steps

- [ ] Integrate real-time data streaming
- [ ] Add user authentication
- [ ] Power BI report templates
- [ ] Export capabilities (PDF, Excel)
