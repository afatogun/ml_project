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


def normalize_tag(value: Optional[str]) -> Optional[str]:
    """
    Normalize a tag: strip whitespace, lowercase.
    Returns None for null/empty values.
    Only "garage" is currently a valid tag.
    """
    if pd.isna(value) or value is None:
        return None
    value = str(value).strip().lower()
    if not value:
        return None
    # Only "garage" is a valid tag currently
    if value == "garage":
        return "garage"
    return None


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load training data (XLSX or CSV) and normalize labels.

    Expected columns: id, Sector, subcategory, type, Description
    Optional columns: tag
    """
    # Support both CSV and XLSX
    if data_path.lower().endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

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

    # Handle optional tag column
    if "tag" in df.columns:
        df["tag"] = df["tag"].apply(normalize_tag)
    else:
        df["tag"] = None

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


def get_eligible_tag_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return boolean mask for rows eligible for tag prediction.
    Eligible: tag is not null (currently only "garage" is valid)
    """
    return df["tag"].notna()


def load_inference_data(data_path: str) -> pd.DataFrame:
    """
    Load inference data (XLSX or CSV) with columns: id, description
    """
    # Support both CSV and XLSX
    if data_path.lower().endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"id", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean description
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    return df


def apply_translation(
    df: pd.DataFrame,
    text_column: str = "description",
    client=None,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """
    Apply Irish-to-English translation to a text column.
    
    Adds three columns to the dataframe:
    - {text_column}_original: The original text before translation
    - {text_column}_translated: The text after translation (same as original if no translation)
    - translation_applied: Boolean flag indicating if translation was performed
    
    The original {text_column} is updated with the translated text for downstream processing.
    
    Args:
        df: DataFrame with text column
        text_column: Name of the column to translate
        client: Optional OpenAI client
        model: OpenAI model for translation
        
    Returns:
        DataFrame with translation columns added
    """
    from .openai_embeddings import batch_translate_irish_to_english, get_client
    
    if client is None:
        client = get_client()
    
    texts = df[text_column].tolist()
    results = batch_translate_irish_to_english(texts, client, model)
    
    # Add translation columns
    df = df.copy()
    df[f"{text_column}_original"] = [r.original_text for r in results]
    df["translation_applied"] = [r.translation_applied for r in results]
    # Update the text column with translated text
    df[text_column] = [r.text for r in results]
    
    return df
