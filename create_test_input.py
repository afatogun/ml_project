#!/usr/bin/env python3
"""
Create a test input file from the training data's holdout set.

This extracts the 10% test split and saves:
1. Input file (id, description only) for prediction
2. Ground truth file (all columns) for validation

Usage:
    python create_test_input.py --train_xlsx ./data/raw/2025_classification_training_set.xlsx
"""

import argparse
import os
import sys

import numpy as np

from src.preprocess import load_training_data
from src.train_models import stratified_split, SEED


def main():
    parser = argparse.ArgumentParser(description="Create test input from training data")
    parser.add_argument(
        "--train_xlsx",
        required=True,
        help="Path to training XLSX file",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.10,
        help="Fraction of data to use as test set (default: 0.10)",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/test",
        help="Directory to save test files",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    np.random.seed(SEED)

    print("=" * 60)
    print("CREATING TEST INPUT FILES")
    print("=" * 60)

    # Load training data
    print(f"\n[1/3] Loading training data from: {args.train_xlsx}")
    df = load_training_data(args.train_xlsx)
    print(f"  Total rows: {len(df)}")

    # Create dummy embeddings (we only need the split indices)
    print(f"\n[2/3] Creating stratified split with test_size={args.test_size}...")
    dummy_embeddings = np.zeros((len(df), 1))  # Minimal dummy for split
    splits = stratified_split(df, dummy_embeddings, test_size=args.test_size)

    test_df = splits["test"]["df"]
    print(f"  Test set size: {len(test_df)} rows")

    # Save files
    print(f"\n[3/3] Saving test files to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Input file (id, description only)
    input_file = os.path.join(args.output_dir, "test_input.xlsx")
    test_df[["id", "description"]].to_excel(input_file, index=False)
    print(f"  Input file: {input_file}")

    # Ground truth file (all columns for validation)
    truth_file = os.path.join(args.output_dir, "test_ground_truth.xlsx")
    test_df.to_excel(truth_file, index=False)
    print(f"  Ground truth file: {truth_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test samples: {len(test_df)}")
    print(f"  Sectors: {test_df['sector'].nunique()}")
    print(f"  Subcategories: {test_df['subcategory'].nunique()} (non-null: {test_df['subcategory'].notna().sum()})")
    print(f"  Types: {test_df['type'].nunique()} (non-null: {test_df['type'].notna().sum()})")

    print("\nNext steps:")
    print(f"  1. Run predictions:")
    print(f"     python predict_xlsx.py --model_dir ./artifacts/<run_id> --input_xlsx {input_file} --output_xlsx ./data/test/test_predictions.xlsx")
    print(f"  2. Validate results:")
    print(f"     python validate_predictions.py --predictions_xlsx ./data/test/test_predictions.xlsx --truth_xlsx {truth_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
