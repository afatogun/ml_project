#!/usr/bin/env python3
"""
Validate predictions against ground truth.

Compares model predictions to known labels and reports accuracy metrics.

Usage:
    python validate_predictions.py --predictions_xlsx ./data/test/test_predictions.xlsx --truth_xlsx ./data/test/test_ground_truth.xlsx
"""

import argparse
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def load_and_merge(predictions_path: str, truth_path: str) -> pd.DataFrame:
    """Load and merge predictions with ground truth."""
    pred_df = pd.read_excel(predictions_path)
    truth_df = pd.read_excel(truth_path)

    # Normalize column names
    pred_df.columns = [c.strip().lower() for c in pred_df.columns]
    truth_df.columns = [c.strip().lower() for c in truth_df.columns]

    # Merge on id
    merged = pred_df.merge(
        truth_df[["id", "sector", "subcategory", "type"]],
        on="id",
        suffixes=("_pred", "_true"),
    )

    return merged


def evaluate_column(
    df: pd.DataFrame,
    pred_col: str,
    true_col: str,
    name: str,
    eligible_mask: pd.Series = None,
) -> dict:
    """Evaluate predictions for a single column."""
    if eligible_mask is not None:
        df_eval = df[eligible_mask].copy()
    else:
        df_eval = df.copy()

    # Handle null predictions
    df_eval = df_eval.dropna(subset=[true_col])

    y_true = df_eval[true_col].astype(str)
    y_pred = df_eval[pred_col].fillna("NULL").astype(str)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Count nulls in predictions
    null_preds = (df_eval[pred_col].isna()).sum()

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": f1,
        "total": len(df_eval),
        "correct": (y_true == y_pred).sum(),
        "null_predictions": null_preds,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_results(results: dict, verbose: bool = False):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {results['name'].upper()}")
    print(f"{'='*60}")
    print(f"  Total samples:     {results['total']}")
    print(f"  Correct:           {results['correct']}")
    print(f"  Null predictions:  {results['null_predictions']} (below threshold)")
    print(f"  Accuracy:          {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Macro-F1:          {results['macro_f1']:.4f}")

    if verbose:
        print(f"\n  Classification Report:")
        print(classification_report(results['y_true'], results['y_pred'], zero_division=0))


def main():
    parser = argparse.ArgumentParser(description="Validate predictions against ground truth")
    parser.add_argument(
        "--predictions_xlsx",
        required=True,
        help="Path to predictions XLSX file",
    )
    parser.add_argument(
        "--truth_xlsx",
        required=True,
        help="Path to ground truth XLSX file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed classification reports",
    )
    parser.add_argument(
        "--output_csv",
        help="Optional: Save detailed comparison to CSV",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PREDICTION VALIDATION")
    print("=" * 60)

    # Load and merge data
    print(f"\nLoading predictions: {args.predictions_xlsx}")
    print(f"Loading ground truth: {args.truth_xlsx}")
    df = load_and_merge(args.predictions_xlsx, args.truth_xlsx)
    print(f"Merged rows: {len(df)}")

    # Evaluate Sector (always predicted)
    sector_results = evaluate_column(
        df,
        pred_col="pred_sector",
        true_col="sector",
        name="Sector",
    )
    print_results(sector_results, args.verbose)

    # Evaluate Subcategory (only where true subcategory exists and sector != Miscellaneous)
    subcat_mask = (
        df["subcategory"].notna() &
        (df["sector"].str.lower() != "miscellaneous")
    )
    if subcat_mask.sum() > 0:
        subcat_results = evaluate_column(
            df,
            pred_col="pred_subcategory",
            true_col="subcategory",
            name="Subcategory",
            eligible_mask=subcat_mask,
        )
        print_results(subcat_results, args.verbose)
    else:
        print("\n  SUBCATEGORY: No eligible samples")

    # Evaluate Type (only where true type exists)
    type_mask = df["type"].notna()
    if type_mask.sum() > 0:
        type_results = evaluate_column(
            df,
            pred_col="pred_type",
            true_col="type",
            name="Type",
            eligible_mask=type_mask,
        )
        print_results(type_results, args.verbose)
    else:
        print("\n  TYPE: No eligible samples")

    # Save detailed comparison if requested
    if args.output_csv:
        comparison_df = df[[
            "id", "description",
            "pred_sector", "sector",
            "pred_subcategory", "subcategory",
            "pred_type", "type",
            "pred_sector_conf", "pred_subcategory_conf", "pred_type_conf",
            "notes",
        ]].copy()

        # Add match columns
        comparison_df["sector_match"] = comparison_df["pred_sector"] == comparison_df["sector"]
        comparison_df["subcategory_match"] = comparison_df["pred_subcategory"] == comparison_df["subcategory"]
        comparison_df["type_match"] = comparison_df["pred_type"] == comparison_df["type"]

        comparison_df.to_csv(args.output_csv, index=False)
        print(f"\nDetailed comparison saved to: {args.output_csv}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Sector Accuracy:      {sector_results['accuracy']*100:.2f}%")
    if subcat_mask.sum() > 0:
        print(f"  Subcategory Accuracy: {subcat_results['accuracy']*100:.2f}%")
    if type_mask.sum() > 0:
        print(f"  Type Accuracy:        {type_results['accuracy']*100:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
