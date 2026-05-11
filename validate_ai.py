"""
Validation and comparison for pure-AI sector + garage tag predictions.

Merges predictions with ground truth, computes metrics, generates
comparison CSV with per-row details and confusion matrices.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def evaluate_column(pred: pd.Series, true: pd.Series, col_name: str) -> dict:
    """
    Compute accuracy and F1-score for a single column.

    Args:
        pred: Predicted values.
        true: Ground truth values.
        col_name: Column name for logging.

    Returns:
        Dictionary with accuracy and F1 (macro).
    """
    # Normalize to strings and keep rows with non-null ground truth.
    mask = pd.notna(true)
    pred_valid = pred[mask].fillna("__MISSING__").astype(str)
    true_valid = true[mask].fillna("__MISSING__").astype(str)

    if len(true_valid) == 0:
        logger.warning(f"{col_name}: No valid ground truth values")
        return {"accuracy": np.nan, "f1_macro": np.nan}

    # Accuracy
    matches = (pred_valid == true_valid).sum()
    accuracy = matches / len(true_valid)

    # F1 macro (simplified: for binary, compute per-class F1)
    unique_vals = set(true_valid.unique()) | set(pred_valid.unique())
    f1_scores = []

    for val in unique_vals:
        tp = ((pred_valid == val) & (true_valid == val)).sum()
        fp = ((pred_valid == val) & (true_valid != val)).sum()
        fn = ((pred_valid != val) & (true_valid == val)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    f1_macro = np.mean(f1_scores) if f1_scores else 0

    return {"accuracy": accuracy, "f1_macro": f1_macro}


def confusion_matrix(pred: pd.Series, true: pd.Series, categories: list = None) -> pd.DataFrame:
    """
    Generate confusion matrix for predicted vs true values.

    Args:
        pred: Predicted values.
        true: Ground truth values.
        categories: Valid categories (to ensure full matrix). If None, inferred.

    Returns:
        DataFrame confusion matrix (rows=true, cols=pred).
    """
    mask = pd.notna(true)
    pred_valid = pred[mask].fillna("__MISSING__").astype(str)
    true_valid = true[mask].fillna("__MISSING__").astype(str)

    if categories is None:
        categories = sorted(set(true_valid.unique()) | set(pred_valid.unique()))
    else:
        categories = [str(c) for c in categories]

    cm = pd.crosstab(true_valid, pred_valid, rownames=["True"], colnames=["Pred"])

    # Reindex to include all categories (fill missing with 0)
    cm = cm.reindex(index=categories, columns=categories, fill_value=0)

    return cm


def validate_predictions(
    predictions_path: str,
    truth_path: str,
    output_dir: str = "reports",
    run_id: str = None,
):
    """
    Compare AI predictions with ground truth.

    Generates:
    - comparison_ai.csv: Row-by-row predictions, truth, matches
    - confusion_sector_ai.csv: 10x10 sector confusion matrix
    - confusion_tag_ai.csv: 2x2 garage tag confusion matrix
    - metrics printout

    Args:
        predictions_path: Path to XLSX with AI predictions.
        truth_path: Path to XLSX with ground truth.
        output_dir: Directory for outputs (default "reports").
        run_id: Run identifier for subfolder. If None, auto-generated from timestamp.

    Returns:
        Dictionary with metrics.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading predictions from {predictions_path}")
    pred_df = pd.read_excel(predictions_path)

    logger.info(f"Loading ground truth from {truth_path}")
    truth_df = pd.read_excel(truth_path)

    # Merge on ID (+description when available) to avoid explosion from duplicate IDs.
    if "id" not in pred_df.columns or "id" not in truth_df.columns:
        raise ValueError("Both dataframes must have 'id' column for merging")

    merge_keys = ["id"]
    if "description" in pred_df.columns and "description" in truth_df.columns:
        merge_keys.append("description")

    pred_before = len(pred_df)
    truth_before = len(truth_df)
    pred_df = pred_df.drop_duplicates(subset=merge_keys, keep="first")
    truth_df = truth_df.drop_duplicates(subset=merge_keys, keep="first")
    if len(pred_df) != pred_before:
        logger.warning(f"Dropped {pred_before - len(pred_df)} duplicate prediction rows before merge")
    if len(truth_df) != truth_before:
        logger.warning(f"Dropped {truth_before - len(truth_df)} duplicate truth rows before merge")

    merged = pred_df.merge(truth_df, on=merge_keys, suffixes=("_pred", "_truth"), how="inner")
    logger.info(f"Merged {len(merged)} rows")

    # Identify sector/tag columns from truth
    # Expect: Sector (true), tag (true); predictions have pred_sector, pred_tag
    sector_col_true = None
    tag_col_true = None

    for col in truth_df.columns:
        if col.lower() == "sector":
            sector_col_true = col
        elif col.lower() == "tag":
            tag_col_true = col

    if sector_col_true is None:
        raise ValueError("Ground truth must have 'Sector' column")
    if tag_col_true is None:
        raise ValueError("Ground truth must have 'tag' column")

    # Normalize tag nulls to 'none' for stable binary evaluation.
    merged[tag_col_true] = merged[tag_col_true].fillna("none").astype(str)
    if "pred_tag" in merged.columns:
        merged["pred_tag"] = merged["pred_tag"].fillna("none").astype(str)

    # Normalize sector prediction column for robust metrics.
    if "pred_sector" in merged.columns:
        merged["pred_sector"] = merged["pred_sector"].fillna("__MISSING__").astype(str)

    # Compare sector
    logger.info("Comparing sector predictions...")
    sector_metrics = evaluate_column(
        merged["pred_sector"], merged[sector_col_true], "Sector"
    )
    sector_cm = confusion_matrix(merged["pred_sector"], merged[sector_col_true])

    # Compare tag
    logger.info("Comparing tag predictions...")
    tag_metrics = evaluate_column(
        merged["pred_tag"], merged[tag_col_true], "Tag"
    )
    tag_cm = confusion_matrix(
        merged["pred_tag"], merged[tag_col_true],
        categories=["garage", "none"]
    )

    # Build comparison CSV
    comparison_cols = [
        "id", "description",
        "pred_sector", "pred_sector_conf",
        "pred_tag", "pred_tag_conf",
        "matched_rule_signals",
        sector_col_true, tag_col_true,
        "rationale", "rulebook_version", "model_version",
    ]

    comparison_df = merged[[c for c in comparison_cols if c in merged.columns]].copy()

    # Add match indicators
    comparison_df["sector_match"] = merged["pred_sector"].fillna("__MISSING__").astype(str) == merged[sector_col_true].fillna("__MISSING__").astype(str)
    comparison_df["tag_match"] = merged["pred_tag"].fillna("none").astype(str) == merged[tag_col_true].fillna("none").astype(str)
    comparison_df["exact_match"] = comparison_df["sector_match"] & comparison_df["tag_match"]

    # Save outputs
    comparison_path = output_path / "comparison_ai.csv"
    logger.info(f"Writing comparison CSV to {comparison_path}")
    comparison_df.to_csv(comparison_path, index=False)

    sector_cm_path = output_path / "confusion_sector_ai.csv"
    logger.info(f"Writing sector confusion matrix to {sector_cm_path}")
    sector_cm.to_csv(sector_cm_path)

    tag_cm_path = output_path / "confusion_tag_ai.csv"
    logger.info(f"Writing tag confusion matrix to {tag_cm_path}")
    tag_cm.to_csv(tag_cm_path)

    # Print metrics
    logger.info("\n" + "="*60)
    logger.info("SECTOR METRICS")
    logger.info("="*60)
    logger.info(f"Accuracy: {sector_metrics['accuracy']:.4f}")
    logger.info(f"F1 (macro): {sector_metrics['f1_macro']:.4f}")
    logger.info("\nConfusion Matrix:")
    print(sector_cm)

    logger.info("\n" + "="*60)
    logger.info("TAG (GARAGE) METRICS")
    logger.info("="*60)
    logger.info(f"Accuracy: {tag_metrics['accuracy']:.4f}")
    logger.info(f"F1 (macro): {tag_metrics['f1_macro']:.4f}")
    logger.info("\nConfusion Matrix:")
    print(tag_cm)

    logger.info(f"\nOutputs saved to {output_path}/")

    return {
        "sector": sector_metrics,
        "tag": tag_metrics,
        "run_id": run_id,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate pure-AI predictions")
    parser.add_argument("--predictions", required=True, help="Predictions XLSX")
    parser.add_argument("--truth", required=True, help="Ground truth XLSX")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--run-id", default=None, help="Run ID for subfolder")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    validate_predictions(
        predictions_path=args.predictions,
        truth_path=args.truth,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
