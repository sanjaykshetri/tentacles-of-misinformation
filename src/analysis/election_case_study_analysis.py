"""Main analysis pipeline for election AI misinformation case studies."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data.load_case_studies import load_case_studies
from src.features.case_study_features import (
    add_case_study_scores,
    build_detection_gap_flags,
    summarize_by_country,
    summarize_by_modality,
)
from src.utils.plotting import (
    plot_risk_by_country,
    plot_cases_by_modality,
    plot_detection_gap,
)


def run_analysis(
    input_csv: str = "data/case_studies/election_ai_misinformation_cases.csv",
    output_dir: str = "outputs/election_case_study",
) -> None:
    """
    Run complete election case study analysis pipeline.
    
    Loads data -> Adds features -> Generates summaries -> Creates visualizations.
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV file.
    output_dir : str
        Output directory for results.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("ELECTION AI MISINFORMATION CASE STUDY ANALYSIS")
    print("="*60)
    
    # Load
    print(f"\n[1/5] Loading case studies from: {input_csv}")
    df = load_case_studies(input_csv)
    print(f"      Loaded {len(df)} cases")

    # Features
    print(f"\n[2/5] Computing risk scores and feature flags")
    df = add_case_study_scores(df)
    df = build_detection_gap_flags(df)
    print(f"      Added: modality_complexity_score, cognitive_intensity_score,")
    print(f"             harm_intent_score, spread_score, response_failure_score,")
    print(f"             overall_case_risk_score, risk_band, text_only_detection_gap")

    # Summaries
    print(f"\n[3/5] Generating country and modality summaries")
    country_summary = summarize_by_country(df)
    modality_summary = summarize_by_modality(df)
    print(f"      Countries represented: {len(country_summary)}")
    print(f"      Modality types: {len(modality_summary)}")

    # Save enriched data
    print(f"\n[4/5] Saving enriched data and summaries")
    enriched_path = output_dir_path / "enriched_case_studies.csv"
    df.to_csv(enriched_path, index=False)
    print(f"      > {enriched_path}")

    country_summary_path = output_dir_path / "country_summary.csv"
    country_summary.to_csv(country_summary_path, index=False)
    print(f"      > {country_summary_path}")

    modality_summary_path = output_dir_path / "modality_summary.csv"
    modality_summary.to_csv(modality_summary_path, index=False)
    print(f"      > {modality_summary_path}")

    # Plots
    print(f"\n[5/5] Generating visualizations")
    plot_risk_by_country(country_summary, output_dir_path)
    print(f"      > {output_dir_path / 'risk_by_country.png'}")
    
    plot_cases_by_modality(modality_summary, output_dir_path)
    print(f"      > {output_dir_path / 'cases_by_modality.png'}")
    
    plot_detection_gap(df, output_dir_path)
    print(f"      > {output_dir_path / 'text_only_detection_gap.png'}")

    # Terminal summary
    print("\n" + "="*60)
    print("TOP HIGH-RISK CASES")
    print("="*60)
    top_cases = df.sort_values("overall_case_risk_score", ascending=False)[
        ["case_id", "country", "election_year", "title", "overall_case_risk_score", "risk_band"]
    ].head(10)
    print(top_cases.to_string(index=False))

    print("\n" + "="*60)
    print("COUNTRY SUMMARY (by average risk)")
    print("="*60)
    print(country_summary.to_string(index=False))

    print("\n" + "="*60)
    print("MODALITY SUMMARY")
    print("="*60)
    print(modality_summary.to_string(index=False))

    print("\n" + "="*60)
    print(f"✓ Analysis complete. Outputs saved to: {output_dir_path.resolve()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_analysis()
