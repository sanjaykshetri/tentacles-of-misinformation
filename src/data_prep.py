"""
Data preparation script for FakeNewsNet dataset.
Loads CSV files with article metadata and generates processed parquet.

Note: FakeNewsNet CSV files contain article IDs, URLs, and titles.
Full article text requires running their data collection scripts with Twitter API keys.
For this sprint, we use titles + URL metadata as the primary features.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path("data/raw/fakenewsnet")

def load_articles_from_csv():
    """Load articles from FakeNewsNet CSV files."""
    rows = []
    
    # Load all CSV files
    csv_files = [
        ("politifact_real.csv", "real", "politifact"),
        ("politifact_fake.csv", "fake", "politifact"),
        ("gossipcop_real.csv", "real", "gossipcop"),
        ("gossipcop_fake.csv", "fake", "gossipcop"),
    ]
    
    for csv_file, label, source in csv_files:
        path = BASE_DIR / csv_file
        if path.exists():
            print(f"Loading {csv_file}...")
            df_temp = pd.read_csv(path)
            
            # Map available columns
            df_temp["label"] = label
            df_temp["dataset"] = source
            
            # Keep id, title, url, and label
            if "news_url" in df_temp.columns:
                df_temp = df_temp[["id", "title", "news_url", "label", "dataset"]].copy()
                df_temp.rename(columns={"news_url": "url"}, inplace=True)
            else:
                df_temp = df_temp[["id", "title", "label", "dataset"]].copy()
                df_temp["url"] = ""
            
            rows.append(df_temp)
        else:
            print(f"  File not found: {path}")
    
    if rows:
        df = pd.concat(rows, ignore_index=True)
        print(f"\nLoaded {len(df)} total articles")
        return df
    else:
        raise ValueError("No CSV files found in data/raw/fakenewsnet/")

def clean_data(df):
    """Basic data cleaning."""
    df = df.copy()
    
    # Remove rows with missing title
    df = df.dropna(subset=["title"])
    
    # Strip whitespace
    df["title"] = df["title"].str.strip()
    
    # Remove empty strings
    df = df[df["title"].str.len() > 0]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["title"], keep="first")
    
    print(f"After cleaning: {len(df)} articles")
    return df

def add_features(df):
    """Add basic text features based on title."""
    df = df.copy()
    
    df["title_length"] = df["title"].str.split().str.len()
    df["title_chars"] = df["title"].str.len()
    df["has_url"] = df["url"].notna() & (df["url"].str.len() > 0)
    df["has_url"] = df["has_url"].astype(int)
    
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("FakeNewsNet Data Preparation")
    print("=" * 60)
    
    # Load
    df = load_articles_from_csv()
    print(f"\nInitial shape: {df.shape}")
    
    # Clean
    df = clean_data(df)
    
    # Add features
    df = add_features(df)
    
    # Summary stats
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(df["label"].value_counts())
    print("\nProportion:")
    print(df["label"].value_counts(normalize=True))
    
    print("\n" + "=" * 60)
    print("DATASET BREAKDOWN")
    print("=" * 60)
    print(df["dataset"].value_counts())
    
    print("\n" + "=" * 60)
    print("TEXT STATISTICS")
    print("=" * 60)
    print(df[["title_length", "title_chars"]].describe())
    
    # Save
    output_path = Path("data/processed/articles.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"\n✅ Saved to {output_path}")
    print(f"Shape: {df.shape}")

