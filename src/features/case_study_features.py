"""Feature engineering and scoring for election case studies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_series(s: pd.Series) -> pd.Series:
    """
    Min-max normalize a numeric pandas Series to [0, 1].
    If all values are the same, return zeros.
    
    Parameters
    ----------
    s : pd.Series
        Input series.
    
    Returns
    -------
    pd.Series
        Normalized series in range [0, 1].
    """
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)


def add_case_study_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretable risk and impact scores to case studies.

    Scores include:
    - modality_complexity_score: How technically complex is the attack modality?
    - cognitive_intensity_score: How powerfully does it exploit behavioral vulnerabilities?
    - harm_intent_score: How directly harmful is the intended effect?
    - spread_score: How rapidly and widely did it spread?
    - response_failure_score: How slow/inadequate was the response?
    - overall_case_risk_score: Weighted aggregate (0-1)
    - risk_band: Categorical label (low/moderate/high)
    
    Parameters
    ----------
    df : pd.DataFrame
        Case study dataframe.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with scores added.
    """
    out = df.copy()

    # Modality complexity: video is hardest to detect, text is easiest
    modality_map = {
        "text": 0.30,
        "audio": 0.65,
        "video": 0.80,
        "audio_video": 1.00,
        "multimodal": 1.00,
    }
    out["modality_complexity_score"] = out["modality"].map(modality_map).fillna(0.50)

    # Cognitive intensity: average of four key triggers
    out["cognitive_intensity_score"] = (
        out["cognitive_trigger_fear"].fillna(0)
        + out["cognitive_trigger_authority"].fillna(0)
        + out["cognitive_trigger_urgency"].fillna(0)
        + out["cognitive_trigger_identity"].fillna(0)
    ) / 4.0

    # Harm intent: voter suppression is most serious, impersonation matters, translation manipulation is subtle
    out["harm_intent_score"] = (
        0.35 * out["contains_impersonation"].fillna(0)
        + 0.40 * out["contains_voter_suppression"].fillna(0)
        + 0.15 * out["contains_translation_manipulation"].fillna(0)
        + 0.10 * out["ai_generated"].fillna(0)
    )

    # Spread: combination of reach (size) and speed (virality)
    reach_score = normalize_series(out["estimated_reach"])
    virality_speed_score = 1 - normalize_series(
        out["time_to_viral_hours"].replace(0, np.nan).fillna(1)
    )
    out["spread_score"] = 0.60 * reach_score + 0.40 * virality_speed_score

    # Response failure: was it fact-checked? how fast did response come?
    fact_check_penalty = out["verified_by_fact_checker"].apply(lambda x: 0 if x == 1 else 1)
    response_delay_score = normalize_series(out["response_delay_hours"])
    out["response_failure_score"] = 0.55 * fact_check_penalty + 0.45 * response_delay_score

    # Overall risk: weighted combination
    out["overall_case_risk_score"] = (
        0.20 * out["modality_complexity_score"]
        + 0.20 * out["cognitive_intensity_score"]
        + 0.25 * out["harm_intent_score"]
        + 0.20 * out["spread_score"]
        + 0.15 * out["response_failure_score"]
    )

    out["overall_case_risk_score"] = out["overall_case_risk_score"].round(3)

    # Risk band: categorical
    out["risk_band"] = pd.cut(
        out["overall_case_risk_score"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["low", "moderate", "high"],
    )

    return out


def build_detection_gap_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag where a text-only misinformation pipeline would likely struggle.

    Rationale:
    - If case contains synthetic voice, text-only pipeline is blind
    - If case contains synthetic video, text-only pipeline is blind
    - If case is impersonation in non-text modality, text-only pipeline is blind
    
    Parameters
    ----------
    df : pd.DataFrame
        Case study dataframe with scores.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with text_only_detection_gap flag added.
    """
    out = df.copy()

    out["text_only_detection_gap"] = (
        (
            (out["contains_synthetic_voice"] == 1)
            | (out["contains_synthetic_video"] == 1)
            | (
                (out["contains_impersonation"] == 1)
                & (
                    out["modality"].isin(
                        ["audio", "video", "audio_video", "multimodal"]
                    )
                )
            )
        )
    ).astype(int)

    return out


def summarize_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate case studies by country.
    
    Parameters
    ----------
    df : pd.DataFrame
        Case study dataframe with scores.
    
    Returns
    -------
    pd.DataFrame
        Country-level summary sorted by risk.
    """
    grouped = (
        df.groupby("country", as_index=False)
        .agg(
            n_cases=("case_id", "count"),
            avg_risk=("overall_case_risk_score", "mean"),
            max_risk=("overall_case_risk_score", "max"),
            text_gap_cases=("text_only_detection_gap", "sum"),
            avg_reach=("estimated_reach", "mean"),
        )
        .sort_values(["avg_risk", "n_cases"], ascending=[False, False])
    )
    grouped["avg_risk"] = grouped["avg_risk"].round(3)
    grouped["avg_reach"] = grouped["avg_reach"].round(0)
    return grouped


def summarize_by_modality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate case studies by modality type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Case study dataframe with scores.
    
    Returns
    -------
    pd.DataFrame
        Modality-level summary sorted by risk.
    """
    grouped = (
        df.groupby("modality", as_index=False)
        .agg(
            n_cases=("case_id", "count"),
            avg_risk=("overall_case_risk_score", "mean"),
            avg_spread=("spread_score", "mean"),
            avg_cognitive_intensity=("cognitive_intensity_score", "mean"),
        )
        .sort_values("avg_risk", ascending=False)
    )
    grouped["avg_risk"] = grouped["avg_risk"].round(3)
    grouped["avg_spread"] = grouped["avg_spread"].round(3)
    grouped["avg_cognitive_intensity"] = grouped["avg_cognitive_intensity"].round(3)
    return grouped
