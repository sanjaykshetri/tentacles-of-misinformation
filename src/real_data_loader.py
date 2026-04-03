"""
Real Data Loader
================
Standard API used by all Phase 2-4 notebooks to load the processed
FakeNewsNet dataset instead of synthetic data.

Usage
-----
    from src.real_data_loader import load_articles_df, load_splits, load_feature_matrices
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_PATH  = PROCESSED_DIR / "articles_processed.parquet"
MODEL_DIR     = PROJECT_ROOT / "models"


def load_articles_df(path: Path = PARQUET_PATH) -> pd.DataFrame:
    """
    Load the full processed FakeNewsNet dataset.

    Returns a DataFrame with columns:
        id, title, url, label, dataset, title_length_words,
        title_length_chars, subjectivity, lexical_diversity,
        certainty_terms, hedging_terms, emotional_intensifiers,
        certainty_hedging_ratio, label_num
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}.\n"
            "Run:  python data/pipeline/quickstart.py"
        )
    df = pd.read_parquet(path)
    df["label_num"] = (df["label"] == "fake").astype(int)
    return df


def load_splits(
    test_size: float = 0.2,
    random_state: int = 42,
    path: Path = PARQUET_PATH,
):
    """
    Return stratified train / val DataFrames.

    Returns
    -------
    train_df, val_df : pd.DataFrame
    """
    df = load_articles_df(path)
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df["label_num"], random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_feature_matrices(model_dir: Path = MODEL_DIR):
    """
    Load (or build) behavioral + NLP feature matrices used by fusion notebooks.

    Returns
    -------
    dict with keys: X_beh_train, X_beh_val, X_nlp_train, X_nlp_val,
                    y_train, y_val, train_df, val_df
    """
    required = [
        "X_beh_train.npy", "X_beh_val.npy",
        "X_nlp_train.npy", "X_nlp_val.npy",
        "y_train.npy", "y_val.npy",
    ]
    if all((model_dir / f).exists() for f in required):
        arrs = {f.replace(".npy", ""): np.load(model_dir / f) for f in required}
        # Also load the DataFrames for text access
        train_df, val_df = load_splits()
        return {**arrs, "train_df": train_df, "val_df": val_df}

    return build_feature_matrices(model_dir=model_dir)


def build_feature_matrices(
    model_dir: Path = MODEL_DIR,
    save: bool = True,
) -> dict:
    """
    Build behavioral + NLP feature matrices from real data and save as .npy.

    Behavioral features: linguistic signals extracted by the pipeline
    NLP features: TF-IDF sparse -> truncated SVD (128 dims)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    print("Building feature matrices from real FakeNewsNet data …")
    train_df, val_df = load_splits()
    y_train = train_df["label_num"].values
    y_val   = val_df["label_num"].values

    # --- Behavioral features (from pipeline) ---
    beh_cols = [
        "title_length_words", "title_length_chars",
        "subjectivity", "lexical_diversity",
        "certainty_terms", "hedging_terms",
        "emotional_intensifiers", "certainty_hedging_ratio",
    ]
    X_beh_train = train_df[beh_cols].fillna(0).values.astype(np.float32)
    X_beh_val   = val_df[beh_cols].fillna(0).values.astype(np.float32)

    # --- NLP features: TF-IDF + SVD ---
    print("  Fitting TF-IDF …")
    tfidf = TfidfVectorizer(
        max_features=10_000, ngram_range=(1, 2),
        min_df=3, max_df=0.9, sublinear_tf=True,
    )
    X_tfidf_tr = tfidf.fit_transform(train_df["title"].fillna(""))
    X_tfidf_vl = tfidf.transform(val_df["title"].fillna(""))

    print("  Fitting TruncatedSVD (128 components) …")
    svd = TruncatedSVD(n_components=128, random_state=42)
    X_nlp_train = svd.fit_transform(X_tfidf_tr).astype(np.float32)
    X_nlp_val   = svd.transform(X_tfidf_vl).astype(np.float32)

    result = {
        "X_beh_train": X_beh_train,
        "X_beh_val":   X_beh_val,
        "X_nlp_train": X_nlp_train,
        "X_nlp_val":   X_nlp_val,
        "y_train":     y_train,
        "y_val":       y_val,
        "train_df":    train_df,
        "val_df":      val_df,
    }

    if save:
        model_dir.mkdir(exist_ok=True)
        for key in ["X_beh_train", "X_beh_val", "X_nlp_train", "X_nlp_val", "y_train", "y_val"]:
            np.save(model_dir / f"{key}.npy", result[key])
            print(f"  Saved {key}.npy  shape={result[key].shape}")

    print(f"  Train: {len(train_df):,} articles | Val: {len(val_df):,}")
    print(f"  Behavioral features: {X_beh_train.shape[1]}  NLP features: {X_nlp_train.shape[1]}")
    return result
