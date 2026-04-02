"""Plotting utilities for election case study analysis."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_dir(path: str | Path) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Parameters
    ----------
    path : str | Path
        Output directory path.
    
    Returns
    -------
    Path
        Output directory Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_risk_by_country(
    df_country: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Plot average risk score by country.
    
    Parameters
    ----------
    df_country : pd.DataFrame
        Country summary dataframe.
    output_dir : str | Path
        Output directory.
    
    Returns
    -------
    Path
        Path to saved figure.
    """
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / "risk_by_country.png"

    plot_df = df_country.sort_values("avg_risk", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["country"], plot_df["avg_risk"], color="steelblue")
    plt.xlabel("Average Risk Score", fontsize=11)
    plt.ylabel("Country", fontsize=11)
    plt.title("Average Election AI-Misinformation Risk by Country", fontsize=12, fontweight="bold")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path


def plot_cases_by_modality(
    df_modality: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Plot number of cases by modality type.
    
    Parameters
    ----------
    df_modality : pd.DataFrame
        Modality summary dataframe.
    output_dir : str | Path
        Output directory.
    
    Returns
    -------
    Path
        Path to saved figure.
    """
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / "cases_by_modality.png"

    plot_df = df_modality.sort_values("n_cases", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["modality"], plot_df["n_cases"], color="coral")
    plt.xlabel("Modality", fontsize=11)
    plt.ylabel("Number of Cases", fontsize=11)
    plt.title("Election AI-Misinformation Cases by Modality", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path


def plot_detection_gap(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Plot how many cases would be missed by text-only detection pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full case study dataframe with detection gap flags.
    output_dir : str | Path
        Output directory.
    
    Returns
    -------
    Path
        Path to saved figure.
    """
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / "text_only_detection_gap.png"

    gap_counts = df["text_only_detection_gap"].value_counts().sort_index()

    labels = ["No Gap\n(Catchable by Text-Only)", "Gap Present\n(Likely Missed)"]
    values = [gap_counts.get(0, 0), gap_counts.get(1, 0)]

    plt.figure(figsize=(6, 4))
    colors = ["lightgreen", "salmon"]
    plt.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)
    plt.ylabel("Number of Cases", fontsize=11)
    plt.title("Text-Only Detection Gap in Election Cases", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path
