"""Data preprocessing utilities."""

import hashlib
from typing import Tuple, Optional

import pandas as pd
import numpy as np


def normalize_label(value: Optional[str]) -> Optional[str]:
    """
    Normalize a label: strip whitespace, title case.
    Returns None for null/empty values.
    """
    if pd.isna(value) or value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    # Title case normalization
    return value.title()


def load_training_data(xlsx_path: str) -> pd.DataFrame:
    """
    Load training XLSX and normalize labels.

    Expected columns: id, Sector, subcategory, type, Description
    """
    df = pd.read_excel(xlsx_path)

    # Normalize column names (handle case variations)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"id", "sector", "subcategory", "type", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize labels
    df["sector"] = df["sector"].apply(normalize_label)
    df["subcategory"] = df["subcategory"].apply(normalize_label)
    df["type"] = df["type"].apply(normalize_label)

    # Clean description
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    # Drop rows with missing sector (required)
    df = df.dropna(subset=["sector"]).reset_index(drop=True)

    return df


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic hash of the dataframe for tracking."""
    content = df.to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_eligible_subcategory_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return boolean mask for rows eligible for subcategory prediction.
    Eligible: subcategory is not null AND sector != 'Miscellaneous'
    """
    has_subcategory = df["subcategory"].notna()
    not_misc = df["sector"].str.lower() != "miscellaneous"
    return has_subcategory & not_misc


def get_eligible_type_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return boolean mask for rows eligible for type prediction.
    Eligible: type is not null
    """
    return df["type"].notna()


def load_inference_data(xlsx_path: str) -> pd.DataFrame:
    """
    Load inference XLSX with columns: id, description
    """
    df = pd.read_excel(xlsx_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"id", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean description
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    return df
