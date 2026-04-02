# Phase 2: Election Case Study Analysis Framework

## Overview

Phase 2 adds a **structured, data-driven case study subsystem** for analyzing election AI misinformation across 8 documented real-world cases spanning 8 countries and 3 modalities (audio, video, audio+video).

This framework:
- ✅ Loads and validates case study data
- ✅ Computes 6 interpretable risk dimensions (modality complexity, cognitive intensity, harm intent, spread, response failure, overall risk)
- ✅ Flags text-only detection gaps (multimodal attacks your current NLP pipeline misses)
- ✅ Generates country-level and modality-level summaries
- ✅ Produces publication-ready visualizations
- ✅ Provides expandable foundation for Phase 3+ (audio/video model outputs)

## Files Created

### Data
- `data/case_studies/election_ai_misinformation_cases.csv` — 8 documented cases with 26 features each
- `data/case_studies/case_annotation_template.json` — Template for adding new cases

### Python Modules
```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── load_case_studies.py        # CSV loader + validator
├── features/
│   ├── __init__.py
│   └── case_study_features.py      # Scoring functions (normalize, risk calc, summaries)
├── analysis/
│   ├── __init__.py
│   └── election_case_study_analysis.py  # Main pipeline
└── utils/
    ├── __init__.py
    └── plotting.py                 # Visualization utilities
```

### Notebooks
- `fusion_models/notebooks/06_election_case_study_demo.ipynb` — Interactive demonstration

## Quick Start

### Run Full Analysis

```bash
cd c:\Users\sanja\OneDrive\Documents\GitHub\tentacles-of-misinformation
python -m src.analysis.election_case_study_analysis
```

**Output:**
```
outputs/election_case_study/
├── enriched_case_studies.csv      # Original data + all computed scores
├── country_summary.csv             # Country-level aggregations
├── modality_summary.csv            # Modality-level aggregations
├── risk_by_country.png
├── cases_by_modality.png
└── text_only_detection_gap.png
```

### In a Jupyter Notebook

```python
from src.data.load_case_studies import load_case_studies
from src.features.case_study_features import (
    add_case_study_scores,
    build_detection_gap_flags,
    summarize_by_country,
)

df = load_case_studies("data/case_studies/election_ai_misinformation_cases.csv")
df = add_case_study_scores(df)
df = build_detection_gap_flags(df)

display(summarize_by_country(df))
```

## Feature Definitions

### Risk Scoring Dimensions

Each case is scored across 6 dimensions (0.0 = low, 1.0 = high):

| Score | Definition | Rationale |
|-------|-----------|-----------|
| **modality_complexity_score** | How hard is the modality to detect? | text=0.3, audio=0.65, video=0.8, audio+video=1.0 |
| **cognitive_intensity_score** | How powerfully does it exploit behavioral vulnerabilities? | Average of 4 cognitive triggers (fear, authority, urgency, identity) |
| **harm_intent_score** | How directly does it cause election-level harm? | Voter suppression (0.4) > Impersonation (0.35) > Translation manipulation (0.15) > AI-generated flag (0.1) |
| **spread_score** | How fast and widely did it reach people? | 60% reach + 40% speed-to-viral |
| **response_failure_score** | How inadequate was the fact-checking response? | 55% fact-check penalty + 45% response delay |
| **overall_case_risk_score** | Weighted aggregate | 0.20×modality + 0.20×cognitive + 0.25×harm + 0.20×spread + 0.15×response |

### Binary Flags

| Field | Meaning |
|-------|---------|
| `ai_generated` | Content was synthesized (not just transmitted) |
| `ai_transmitted` | AI was used for amplification/distribution |
| `contains_synthetic_voice` | Voice cloning / TTS detected |
| `contains_synthetic_video` | Deepfake video detected |
| `contains_impersonation` | Impersonated real person/authority |
| `contains_voter_suppression` | Intended to discourage voting |
| `contains_translation_manipulation` | Content manipulated across languages |
| `verified_by_fact_checker` | Fact-checked by professional org |
| `text_only_detection_gap` | **FLAG**: Text-only NLP cannot catch this |

### Cognitive Triggers (0.0–1.0)

From Chapter 1 behavioral analysis. Each case is scored on how much it exploits:
- `cognitive_trigger_fear` — Anxiety, threat perception
- `cognitive_trigger_authority` — Trust in source, credibility exploitation
- `cognitive_trigger_urgency` — Time pressure, recency effect
- `cognitive_trigger_identity` — Tribal/group belonging, in-group bias

## Key Findings from Current Data

### Top Risks
1. **India 2024** (audio+video multimodal) — Risk: 0.668
   - 50+ variants, 3M reach, political party distribution
2. **United States 2024** (audio robocall) — Risk: 0.592
   - Direct voter suppression, FBI investigation
3. **Nepal 2024** (video deepfakes) — Risk: 0.565
   - Exploited language detection gap

### Modality Ranking
- Audio+Video: 0.668 avg risk (hardest to detect)
- Audio: 0.518 avg risk
- Video: 0.517 avg risk

### Detection Gap
- **8 out of 8 cases** (100%) have text-only detection gaps
- All involve synthetic voice, synthetic video, or impersonation in non-text modality
- This is **your portfolio strength**: "I identified where current systems fail"

## Adding New Cases

1. Open `data/case_studies/case_annotation_template.json`
2. Copy template values; fill in your case
3. Add row to `election_ai_misinformation_cases.csv`
4. Re-run `python -m src.analysis.election_case_study_analysis`
5. Scores auto-recompute; outputs update

**Example row to add:**
```csv
9,Argentina,2023,Presidential,Title here,Description,audio,"Platform;list",1,1,voters,intended_effect,12,500000,1,0,0,1,0,0.7,0.8,0.6,0.5,1,18,high_risk,Notes
```

## Integration with Tentacles of Misinformation

### Chapter 5 Usage

**Your chapter now has:**
- Real-world cases backed by data (not just narrative)
- Quantified risk scores  
- Visualizations showing modality breakdown, country trends, detection gaps
- Supporting table: "Cases with text-only detection gap" → "Multimodal models needed"

**Quote for narrative:**
> "Across 8 documented election cases (2023–2024), 100% involved modalities beyond text: synthetic voice, deepfake video, or impersonation. A text-only NLP system would miss every single case. This is why the future of election misinformation detection requires multimodal fusion."

### Streamlit Dashboard (Optional Extension)

You can transform this into an interactive dashboard:
```python
import streamlit as st
df = load_case_studies(...)
df = add_case_study_scores(df)

st.dataframe(df)
st.plotly_express.bar(...)  # etc
```

### Academic Paper (Optional Extension)

The scoring methodology is reproducible and defensible:
- Clear feature definitions
- Normalized 0–1 scales
- Weighted aggregates with documented rationale
- Handles missing data appropriately

## Next Steps (Phase 3+)

### Phase 3: Audio Model Development
- Train audio deepfake detector on ASVspoof, real election data
- Column to add: `audio_forensics_confidence`
- Test on Slovakia, India, US robocall cases

### Phase 4: Video Model Development
- Train video deepfake detector on FaceForensics++, election data
- Column to add: `video_forensics_confidence`
- Test on Taiwan, Nepal, Mexico, Brazil video cases

### Phase 4B: Multimodal Fusion
- Combine `overall_case_risk_score` + audio + video + metadata
- Compare: text-only vs. ensemble detectability
- Quantify improvement

## Files Quick Reference

| File | Purpose | Key Function |
|------|---------|--------------|
| `load_case_studies.py` | Data loading | `load_case_studies(csv_path)` → DataFrame |
| `case_study_features.py` | Scoring | `add_case_study_scores(df)` → DataFrame with scores |
| `case_study_features.py` | Gap detection | `build_detection_gap_flags(df)` → Flag multimodal attacks |
| `case_study_features.py` | Summaries | `summarize_by_country(df)`, `summarize_by_modality(df)` |
| `plotting.py` | Visualization | `plot_risk_by_country(df_country, output_dir)` etc. |
| `election_case_study_analysis.py` | Pipeline | `run_analysis()` runs full 5-step pipeline |

## Architecture Diagram

```
Data Input (CSV)
    ↓
Load & Validate
    ↓
Compute 6 Risk Dimensions
    ↓
Aggregate (Country, Modality)
    ↓
Flagging (Detection Gap)
    ↓
Visualizations + Summary Tables
    ↓
Outputs (CSV + PNG)
```

## Verification

Run:
```bash
python -m src.analysis.election_case_study_analysis
```

Expected output:
```
[1/5] Loading case studies from: data/case_studies/...
      Loaded 8 cases
[2/5] Computing risk scores...
[3/5] Generating summaries...
[4/5] Saving enriched data and summaries...
[5/5] Generating visualizations...
✓ Analysis complete. Outputs saved to: outputs/election_case_study
```

All 6 output files should be present:
- enriched_case_studies.csv
- country_summary.csv
- modality_summary.csv
- risk_by_country.png
- cases_by_modality.png
- text_only_detection_gap.png
