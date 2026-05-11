"""Inference utilities."""

import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from .openai_embeddings import batch_semantic_manual_review


# Valid sector → subcategory mapping
SECTOR_SUBCATEGORIES = {
    "Agriculture": ["Agricultural Building"],
    "Civil": ["Carpark", "Power Generation", "Quarry", "Road & Path", "Transport", "Water & Sewerage"],
    "Commercial & Retail": ["Bar & Restaurant", "Car Showroom", "Hotel & Guesthouse", "Office", "Retail", "Service Station"],
    "Education": ["Pre School", "School", "University"],
    "Industrial": ["Data Centre", "Factory", "Light Industrial", "Warehouse"],
    "Medical": ["Care Home", "Hospital", "Medical Centre"],
    "Residential": ["Apartments", "Houses", "Mixed Development", "Student Accommodation"],
    "Self Build": ["Alteration", "Extension", "House"],
    "Social": ["Church & Community", "Public Building", "Sport & Leisure"],
    "Supply & Services": ["Construction Supplies", "Professional Services"],
}


MISC_SECTOR = "Miscellaneous"
MISC_EQUIVALENTS = {"miscellaneous", "shec"}

# Rule signal groups
STRONG_CONSTRUCTION = [
    r"\bconstruction\b",
    r"\berection\b",
    r"\bnew\s+build\b",
    r"\bnew\s+dwelling\b",
    r"\bbuild(ing)?\b",
    r"\binstallation\b",
]

WEAK_CONSTRUCTION = [
    r"\bdevelopment\b",
    r"\bworks\b",
    r"\bextension\b",
]

RETENTION = [
    r"\bretention\b",
    r"\bretention\s+of\b",
    r"\bpermission\s+for\s+retention\b",
]

VARIATION = [
    r"\bvariation\s+of\s+condition\b",
    r"\bvary\s+condition\b",
    r"\bamend\s+condition\b",
]

PLANNING_REF_STRONG = [
    r"\bplanning\s*ref(?:erence)?\s*[:\-]?\s*\w+",
    r"\bref(?:erence)?\s*no\.?\s*[:\-]?\s*\w+",
    r"\breg(?:istration)?\s*no\.?\s*[:\-]?\s*\w+",
]

PLANNING_REF_WEAK = [
    r"\bunder\s*ref(?:erence)?\b",
    r"\bplanning\s*ref(?:erence)?\b",
]

NON_CONSTRUCTION = [
    r"\bchange\s+of\s+use\b",
    r"\bretention\b",
    r"\bpermission\s+for\s+retention\b",
    r"\bregulari[sz]ation\b",
    r"\bamendment\b",
    r"\bmodification\b",
    r"\bextension\s+of\s+duration\b",
    r"\boutline\s+permission\b(?!.*construction)",
]

PRIOR_PERMISSION_SIGNALS = [
    r"\bpreviously\s+approved\b",
    r"\bapproved\s+under\b",
    r"\bgranted\s+permission\b",
    r"\bmaterial\s+amendment\b",
    r"\bamendment\s+to\b",
    r"\bamendments\s+to\b",
    r"\bchange\s+of\s+house\s+type\b",
    r"\bvary\s+condition\b",
    r"\bvariation\s+of\s+condition\b",
    r"\bsection\s+54\b",
    r"\bretention\s+permission\b",
    r"\bsubject\s+to\s+condition\b",
    r"\bcondition\s+no\b",
    r"\bin\s+accordance\s+with\s+approved\b",
]


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


def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


def _matches_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _is_misc_equivalent(sector: Optional[str]) -> bool:
    if sector is None:
        return False
    return str(sector).strip().lower() in MISC_EQUIVALENTS


def should_allow_misc_override(rule: str, sector_confidence: float, has_strong_construction: bool) -> bool:
    high_confidence_rules = {"variation", "retention_only"}

    if has_strong_construction:
        return False

    if rule in high_confidence_rules:
        return True

    if sector_confidence < 0.75:
        return True

    return False


def apply_misc_rules(
    description: str,
    sector_pred: Optional[str],
    sector_confidence: float,
) -> Tuple[Optional[str], bool, Optional[str], List[str]]:
    """
    Apply deterministic Miscellaneous/SHEC rules after sector prediction.

    Order:
      1) planning reference detection
      2) variation of condition
      3) retention handling
      4) no direct construction
    """
    text = _normalize_text(description)
    notes: List[str] = []
    manual_review = False
    manual_review_reason: Optional[str] = None

    has_strong_construction = _matches_any(text, STRONG_CONSTRUCTION)
    has_weak_construction = _matches_any(text, WEAK_CONSTRUCTION)
    construction_present = has_strong_construction or has_weak_construction
    has_retention = _matches_any(text, RETENTION)
    has_variation = _matches_any(text, VARIATION)
    has_non_construction = _matches_any(text, NON_CONSTRUCTION)

    # 1) Manual review detection (deterministic prior permission references)
    if _matches_any(text, PRIOR_PERMISSION_SIGNALS):
        manual_review = True
        manual_review_reason = "prior_permission_reference"
        notes.append("flagged: prior permission reference detected")
    elif _matches_any(text, PLANNING_REF_STRONG):
        manual_review = True
        manual_review_reason = "planning_reference_detected"
        notes.append("flagged: planning reference detected")
    elif _matches_any(text, PLANNING_REF_WEAK) and not has_strong_construction:
        manual_review = True
        manual_review_reason = "planning_reference_detected"
        notes.append("flagged: planning reference detected")

    # 2) Variation of condition -> Miscellaneous when gate allows
    if has_variation:
        if should_allow_misc_override("variation", sector_confidence, has_strong_construction):
            sector_pred = MISC_SECTOR
            notes.append("override: variation of condition → miscellaneous")
        else:
            notes.append("override skipped: high model confidence")

    # 3) Retention handling
    if has_retention and not has_strong_construction:
        if should_allow_misc_override("retention_only", sector_confidence, False):
            sector_pred = MISC_SECTOR
            notes.append("override: retention only → miscellaneous")
        else:
            notes.append("override skipped: high model confidence")
    elif has_retention and has_strong_construction:
        notes.append("retention ignored due to construction")

    # 4) No direct construction -> Miscellaneous when gate allows
    if has_non_construction and not construction_present:
        if should_allow_misc_override("no_construction", sector_confidence, False):
            sector_pred = MISC_SECTOR
            notes.append("override: no direct construction → miscellaneous")
        else:
            notes.append("override skipped: high model confidence")

    return sector_pred, manual_review, manual_review_reason, notes


def load_artifacts(model_dir: str) -> Dict[str, Any]:
    """
    Load all artifacts from a model directory.
    """
    # Load metadata
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    artifacts = {"metadata": metadata}

    # Load classifiers
    for name in ["sector", "subcategory", "type", "tag"]:
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


def predict_constrained_subcategory(
    proba_row: np.ndarray,
    encoder,
    valid_subcategories: List[str],
) -> Tuple[Optional[str], float]:
    """
    Pick the highest-confidence subcategory that is valid for the given sector.

    Args:
        proba_row: 1-D probability vector for one sample
        encoder: LabelEncoder used for subcategory
        valid_subcategories: list of allowed subcategory names for this sector

    Returns: (best_valid_subcategory or None, confidence)
    """
    all_classes = encoder.classes_
    # Build mask of valid class indices
    valid_indices = [i for i, cls in enumerate(all_classes) if cls in valid_subcategories]
    if not valid_indices:
        return None, 0.0

    # Get probabilities only for valid classes
    valid_probs = proba_row[valid_indices]
    best_local = np.argmax(valid_probs)
    best_idx = valid_indices[best_local]
    return all_classes[best_idx], float(proba_row[best_idx])


def run_inference(
    embeddings: np.ndarray,
    artifacts: Dict[str, Any],
    descriptions: List[str] = None,
    subcategory_threshold: float = 0.60,
    type_threshold: float = 0.60,
    tag_threshold: float = 0.60,
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
        tag_threshold: Confidence threshold for tag
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
        "pred_tag": [None] * n,
        "pred_tag_conf": [0.0] * n,
        "manual_review": [False] * n,
        "manual_review_reason": [None] * n,
        "ai_manual_review_confidence": [None] * n,
        "ai_manual_review_rationale": [None] * n,
        "notes": [""] * n,
    }

    # 1. Sector prediction (always)
    sector_preds, sector_confs = predict_with_confidence(
        artifacts["sector"]["model"],
        artifacts["sector"]["encoder"],
        embeddings,
    )

    # Apply deterministic rule engine and post-processing overrides if descriptions provided
    if apply_overrides and descriptions is not None:
        ai_review_results = batch_semantic_manual_review(
            descriptions,
            model="gpt-4o-mini",
        )

        for i in range(n):
            original_sector = sector_preds[i]
            updated_sector, manual_review, manual_review_reason, rule_notes = apply_misc_rules(
                descriptions[i],
                sector_preds[i],
                float(sector_confs[i]),
            )

            ai_result = ai_review_results[i]

            sector_preds[i] = updated_sector
            results["manual_review"][i] = manual_review
            results["manual_review_reason"][i] = manual_review_reason
            results["ai_manual_review_confidence"][i] = round(ai_result.confidence, 4)
            results["ai_manual_review_rationale"][i] = ai_result.rationale

            # AI semantic escalation runs on all rows; deterministic reasons take precedence.
            if ai_result.manual_review:
                if not results["manual_review"][i]:
                    results["manual_review"][i] = True
                    if ai_result.reason:
                        results["manual_review_reason"][i] = ai_result.reason
                    ai_note = "flagged: ai semantic ambiguity detected"
                    rule_notes.append(ai_note)
                elif not results["manual_review_reason"][i] and ai_result.reason:
                    results["manual_review_reason"][i] = ai_result.reason

            if rule_notes:
                results["notes"][i] = "; ".join(rule_notes)

            # Preserve existing heuristic Misc correction only when no deterministic override changed sector.
            if updated_sector == original_sector:
                corrected_sector, override_note = apply_sector_override(
                    descriptions[i],
                    sector_preds[i],
                    sector_confs[i],
                )
                if corrected_sector != sector_preds[i]:
                    sector_preds[i] = corrected_sector
                    if override_note:
                        results["notes"][i] = (
                            override_note
                            if not results["notes"][i]
                            else results["notes"][i] + "; " + override_note
                        )

    results["pred_sector"] = sector_preds
    results["pred_sector_conf"] = np.round(sector_confs, 4)

    # 2. Subcategory prediction (gated, constrained to sector)
    if artifacts["subcategory"]["model"] is not None:
        subcat_model = artifacts["subcategory"]["model"]
        subcat_encoder = artifacts["subcategory"]["encoder"]
        subcat_proba = subcat_model.predict_proba(embeddings)

        for i in range(n):
            notes = []
            sector = sector_preds[i]

            # Gate: if sector is Miscellaneous or not in the mapping, skip subcategory
            if not sector or _is_misc_equivalent(sector) or sector not in SECTOR_SUBCATEGORIES:
                results["pred_subcategory"][i] = None
                results["pred_subcategory_conf"][i] = 0.0
                if sector and _is_misc_equivalent(sector):
                    notes.append("Subcategory skipped: Sector is Miscellaneous")
                elif sector:
                    notes.append(f"Subcategory skipped: no valid subcategories for {sector}")
            else:
                # Constrained prediction: only pick from valid subcategories for this sector
                valid_subs = SECTOR_SUBCATEGORIES[sector]
                best_sub, best_conf = predict_constrained_subcategory(
                    subcat_proba[i], subcat_encoder, valid_subs
                )

                if best_sub is None or best_conf < subcategory_threshold:
                    results["pred_subcategory"][i] = None
                    results["pred_subcategory_conf"][i] = round(best_conf, 4) if best_sub else 0.0
                    notes.append(f"Subcategory below threshold ({best_conf:.2f} < {subcategory_threshold})")
                else:
                    results["pred_subcategory"][i] = best_sub
                    results["pred_subcategory_conf"][i] = round(best_conf, 4)

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

    # 4. Tag prediction (gated by confidence, only predict "garage" when confident)
    if artifacts.get("tag") and artifacts["tag"]["model"] is not None:
        tag_model = artifacts["tag"]["model"]
        tag_encoder = artifacts["tag"]["encoder"]

        # Guardrail: if tag model was trained on a single class ("garage" only),
        # it will predict garage for every row. Fall back to keyword-based tagging.
        is_single_class_tag_model = (
            tag_encoder is not None
            and hasattr(tag_encoder, "classes_")
            and len(tag_encoder.classes_) == 1
            and str(tag_encoder.classes_[0]).lower() == "garage"
        )

        if is_single_class_tag_model:
            garage_pattern = re.compile(r"\bgarage\b", flags=re.IGNORECASE)
            for i in range(n):
                desc = descriptions[i] if descriptions is not None else ""
                if garage_pattern.search(desc or ""):
                    results["pred_tag"][i] = "garage"
                    results["pred_tag_conf"][i] = 1.0
                else:
                    results["pred_tag"][i] = None
                    results["pred_tag_conf"][i] = 0.0
        else:
            tag_preds, tag_confs = predict_with_confidence(
                tag_model,
                tag_encoder,
                embeddings,
            )

            for i in range(n):
                # Only predict "garage" if confidence is above threshold
                # and the model actually predicts "garage"
                if tag_preds[i] == "garage" and tag_confs[i] >= tag_threshold:
                    results["pred_tag"][i] = "garage"
                    results["pred_tag_conf"][i] = round(tag_confs[i], 4)
                else:
                    results["pred_tag"][i] = None
                    results["pred_tag_conf"][i] = round(tag_confs[i], 4) if tag_confs[i] > 0 else 0.0
                    if tag_preds[i] == "garage" and tag_confs[i] < tag_threshold:
                        note = f"Tag below threshold ({tag_confs[i]:.2f} < {tag_threshold})"
                        results["notes"][i] = note if not results["notes"][i] else results["notes"][i] + "; " + note

    return pd.DataFrame(results)
