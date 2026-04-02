"""Load and validate election AI misinformation case studies."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_case_studies(csv_path: str | Path) -> pd.DataFrame:
    """
    Load election AI misinformation case studies from CSV.

    Parameters
    ----------
    csv_path : str | Path
        Path to the case study CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
        
    Raises
    ------
    FileNotFoundError
        If CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    expected_columns = {
        "case_id",
        "country",
        "election_year",
        "election_type",
        "title",
        "description",
        "modality",
        "platforms",
        "ai_generated",
        "ai_transmitted",
        "target_actor",
        "intended_effect",
        "time_to_viral_hours",
        "estimated_reach",
        "contains_impersonation",
        "contains_voter_suppression",
        "contains_translation_manipulation",
        "contains_synthetic_voice",
        "contains_synthetic_video",
        "cognitive_trigger_fear",
        "cognitive_trigger_authority",
        "cognitive_trigger_urgency",
        "cognitive_trigger_identity",
        "verified_by_fact_checker",
        "response_delay_hours",
        "outcome_label",
        "notes",
    }

    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Normalize simple binary fields
    binary_cols = [
        "ai_generated",
        "ai_transmitted",
        "contains_impersonation",
        "contains_voter_suppression",
        "contains_translation_manipulation",
        "contains_synthetic_voice",
        "contains_synthetic_video",
        "verified_by_fact_checker",
    ]
    for col in binary_cols:
        df[col] = df[col].astype(int)

    # Numeric fields
    numeric_cols = [
        "election_year",
        "time_to_viral_hours",
        "estimated_reach",
        "cognitive_trigger_fear",
        "cognitive_trigger_authority",
        "cognitive_trigger_urgency",
        "cognitive_trigger_identity",
        "response_delay_hours",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize text
    text_cols = [
        "country",
        "election_type",
        "title",
        "description",
        "modality",
        "platforms",
        "target_actor",
        "intended_effect",
        "outcome_label",
        "notes",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df
