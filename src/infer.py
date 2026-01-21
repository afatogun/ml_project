"""Inference utilities."""

import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib


# Post-processing rules to correct obvious Miscellaneous misclassifications
# Each rule: (sector_override, keywords_must_have, keywords_must_not_have)
SECTOR_OVERRIDE_RULES = [
    # Self Build: dwelling/house indicators (not multi-unit)
    {
        "sector": "Self Build",
        "must_have": [r"\b(dwelling|house|bungalow|cottage)\b"],
        "must_not_have": [r"\b(change of use|apartments|units|\d+\s*no\.?\s*(dwelling|house|unit)s?)\b"],
        "description": "Single dwelling detected",
    },
    # Residential: multiple dwellings/apartments
    {
        "sector": "Residential",
        "must_have": [r"\b(\d+\s*no\.?\s*(dwelling|house|unit|apartment)s?|apartments|multiple\s+dwelling)\b"],
        "must_not_have": [r"\b(change of use from|retention of)\b"],
        "description": "Multiple residential units detected",
    },
    # Social: museum, church, community, sport facilities
    {
        "sector": "Social",
        "must_have": [r"\b(museum|church|community\s+(centre|hall)|sports?\s*(club|facility|ground)|gaa|rugby|soccer|football\s+club)\b"],
        "must_not_have": [],
        "description": "Social/community facility detected",
    },
    # Education: school, college, university
    {
        "sector": "Education",
        "must_have": [r"\b(school|college|university|classroom|education)\b"],
        "must_not_have": [],
        "description": "Education facility detected",
    },
    # Civil: carpark, road infrastructure
    {
        "sector": "Civil",
        "must_have": [r"\b(carpark|car\s*park|road\s+(construction|widening)|roundabout|junction\s+improvement)\b"],
        "must_not_have": [],
        "description": "Civil infrastructure detected",
    },
    # Agriculture: farm, agricultural
    {
        "sector": "Agriculture",
        "must_have": [r"\b(farm\s*(building|shed)|agricultural\s*(shed|building)|cattle|slatted|silage|hay\s*barn)\b"],
        "must_not_have": [],
        "description": "Agricultural facility detected",
    },
]


def apply_sector_override(description: str, predicted_sector: str, confidence: float) -> Tuple[str, str]:
    """
    Apply post-processing rules to correct obvious Miscellaneous misclassifications.
    
    Returns: (corrected_sector, override_note or empty string)
    """
    if predicted_sector != "Miscellaneous":
        return predicted_sector, ""
    
    desc_lower = description.lower()
    
    for rule in SECTOR_OVERRIDE_RULES:
        # Check must_have patterns (any match)
        has_required = any(re.search(pattern, desc_lower) for pattern in rule["must_have"])
        
        if not has_required:
            continue
        
        # Check must_not_have patterns (none should match)
        has_excluded = any(re.search(pattern, desc_lower) for pattern in rule["must_not_have"])
        
        if has_excluded:
            continue
        
        # Rule matches - override the prediction
        return rule["sector"], f"Override: {rule['description']}"
    
    return predicted_sector, ""


def load_artifacts(model_dir: str) -> Dict[str, Any]:
    """
    Load all artifacts from a model directory.
    """
    # Load metadata
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    artifacts = {"metadata": metadata}

    # Load classifiers
    for name in ["sector", "subcategory", "type"]:
        model_path = os.path.join(model_dir, f"{name}_model.joblib")
        encoder_path = os.path.join(model_dir, f"{name}_encoder.joblib")

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            artifacts[name] = {
                "model": joblib.load(model_path),
                "encoder": joblib.load(encoder_path),
            }
        else:
            artifacts[name] = {"model": None, "encoder": None}

    return artifacts


def predict_with_confidence(
    model,
    encoder,
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict labels and confidence scores.

    Returns: (predictions, confidences)
    """
    if model is None:
        return np.array([None] * len(embeddings)), np.array([0.0] * len(embeddings))

    proba = model.predict_proba(embeddings)
    pred_idx = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    predictions = encoder.inverse_transform(pred_idx)

    return predictions, confidence


def run_inference(
    embeddings: np.ndarray,
    artifacts: Dict[str, Any],
    descriptions: List[str] = None,
    subcategory_threshold: float = 0.60,
    type_threshold: float = 0.60,
    apply_overrides: bool = True,
) -> pd.DataFrame:
    """
    Run gated hierarchical inference.
    
    Args:
        embeddings: Feature embeddings
        artifacts: Loaded model artifacts
        descriptions: Original text descriptions (for post-processing rules)
        subcategory_threshold: Confidence threshold for subcategory
        type_threshold: Confidence threshold for type
        apply_overrides: Whether to apply post-processing rules for Misc corrections

    Returns DataFrame with prediction columns.
    """
    n = len(embeddings)

    # Initialize output columns
    results = {
        "pred_sector": [None] * n,
        "pred_sector_conf": [0.0] * n,
        "pred_subcategory": [None] * n,
        "pred_subcategory_conf": [0.0] * n,
        "pred_type": [None] * n,
        "pred_type_conf": [0.0] * n,
        "notes": [""] * n,
    }

    # 1. Sector prediction (always)
    sector_preds, sector_confs = predict_with_confidence(
        artifacts["sector"]["model"],
        artifacts["sector"]["encoder"],
        embeddings,
    )
    
    # Apply post-processing overrides if descriptions provided
    if apply_overrides and descriptions is not None:
        for i in range(n):
            corrected_sector, override_note = apply_sector_override(
                descriptions[i], 
                sector_preds[i], 
                sector_confs[i]
            )
            if corrected_sector != sector_preds[i]:
                sector_preds[i] = corrected_sector
                results["notes"][i] = override_note
    
    results["pred_sector"] = sector_preds
    results["pred_sector_conf"] = np.round(sector_confs, 4)

    # 2. Subcategory prediction (gated)
    if artifacts["subcategory"]["model"] is not None:
        subcat_preds, subcat_confs = predict_with_confidence(
            artifacts["subcategory"]["model"],
            artifacts["subcategory"]["encoder"],
            embeddings,
        )

        for i in range(n):
            notes = []

            # Gate: if sector is Miscellaneous, skip subcategory
            if sector_preds[i] and sector_preds[i].lower() == "miscellaneous":
                results["pred_subcategory"][i] = None
                results["pred_subcategory_conf"][i] = 0.0
                notes.append("Subcategory skipped: Sector is Miscellaneous")
            # Gate: confidence threshold
            elif subcat_confs[i] < subcategory_threshold:
                results["pred_subcategory"][i] = None
                results["pred_subcategory_conf"][i] = round(subcat_confs[i], 4)
                notes.append(f"Subcategory below threshold ({subcat_confs[i]:.2f} < {subcategory_threshold})")
            else:
                results["pred_subcategory"][i] = subcat_preds[i]
                results["pred_subcategory_conf"][i] = round(subcat_confs[i], 4)

            if notes:
                results["notes"][i] = "; ".join(notes) if not results["notes"][i] else results["notes"][i] + "; " + "; ".join(notes)

    # 3. Type prediction (gated by confidence only)
    if artifacts["type"]["model"] is not None:
        type_preds, type_confs = predict_with_confidence(
            artifacts["type"]["model"],
            artifacts["type"]["encoder"],
            embeddings,
        )

        for i in range(n):
            if type_confs[i] < type_threshold:
                results["pred_type"][i] = None
                results["pred_type_conf"][i] = round(type_confs[i], 4)
                note = f"Type below threshold ({type_confs[i]:.2f} < {type_threshold})"
                results["notes"][i] = note if not results["notes"][i] else results["notes"][i] + "; " + note
            else:
                results["pred_type"][i] = type_preds[i]
                results["pred_type_conf"][i] = round(type_confs[i], 4)

    return pd.DataFrame(results)
