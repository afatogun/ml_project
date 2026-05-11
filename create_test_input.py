#!/usr/bin/env python3
"""
Create a test input file for prediction/validation runs.

Supports two sampling sources:
1. full: random sample from the entire dataset (default)
2. stratified_test: sample from a stratified holdout split (OR use pre-saved indices for reproducibility)

Saves:
1. Input file (id, description only) for prediction
2. Ground truth file (all columns) for validation

Usage:
    python create_test_input.py --train_xlsx ./data/raw/2025_classification_training_set.xlsx

For reproducible test-set sampling from a supervised model's training split:
    python create_test_input.py --train_xlsx ./data/raw/2025_classification_training_set.xlsx \\
        --sample_source stratified_test --split_indices_path ./artifacts/20260511_151816_supervised/split_indices.npy
"""

import argparse
import os
import sys
import secrets

import numpy as np

from src.preprocess import load_training_data
from src.train_models import stratified_split, SEED


def resolve_train_path(train_xlsx: str) -> str:
    """Resolve training file path with sensible workspace defaults."""
    if os.path.exists(train_xlsx):
        return train_xlsx

    # If user passed only a filename, try common project locations.
    basename = os.path.basename(train_xlsx)
    candidates = [
        os.path.join(".", "data", "raw", basename),
        os.path.join(".", basename),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find training file '{train_xlsx}'. "
        f"Tried: {', '.join(candidates)}"
    )


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
        help="Fraction of data for stratified holdout when --sample_source stratified_test (default: 0.10)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of projects to sample (default: 1000)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=SEED,
        help="Random seed for sampling reproducibility (default: 42)",
    )
    parser.add_argument(
        "--random_mode",
        action="store_true",
        help="Generate a fresh random seed on each run",
    )
    parser.add_argument(
        "--split_indices_path",
        default=None,
        help="Path to pre-saved split indices (.npy file) for reproducible train/test separation. "
             "When provided with --sample_source stratified_test, uses these exact indices instead of generating new ones.",
    )
    parser.add_argument(
        "--sample_source",
        choices=["full", "stratified_test"],
        default="full",
        help="Sampling pool: full dataset or stratified holdout split (default: full)",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/test",
        help="Directory to save test files",
    )

    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample_size must be > 0")

    effective_random_seed = secrets.randbelow(2**31 - 1) if args.random_mode else args.random_seed

    # Set seed for reproducibility
    np.random.seed(effective_random_seed)

    print("=" * 60)
    print("CREATING TEST INPUT FILES")
    print("=" * 60)
    print(f"  Random mode: {args.random_mode}")
    print(f"  Random seed: {effective_random_seed}")

    # Load training data
    resolved_train_xlsx = resolve_train_path(args.train_xlsx)
    print(f"\n[1/3] Loading training data from: {resolved_train_xlsx}")
    df = load_training_data(resolved_train_xlsx)
    print(f"  Total rows: {len(df)}")

    # Build candidate pool and sample
    if args.sample_source == "stratified_test":
        if args.split_indices_path and os.path.exists(args.split_indices_path):
            print(f"\n[2/3] Loading pre-saved test indices from: {args.split_indices_path}")
            test_indices = np.load(args.split_indices_path)
            pool_df = df.iloc[test_indices].reset_index(drop=True)
            print(f"  Loaded test pool size: {len(pool_df)} rows (from saved 90/10 split)")
            print(f"  ✓ Using EXACT indices from training run for reproducible validation")
        else:
            print(f"\n[2/3] Creating stratified split with test_size={args.test_size}...")
            dummy_embeddings = np.zeros((len(df), 1))  # Minimal dummy for split
            splits = stratified_split(df, dummy_embeddings, test_size=args.test_size)
            pool_df = splits["test"]["df"]
            print(f"  Stratified pool size: {len(pool_df)} rows")
            if args.split_indices_path:
                print(f"  WARNING: --split_indices_path provided but file not found: {args.split_indices_path}")
                print(f"           Generating NEW split instead (test set may contain training data!)")
    else:
        print(f"\n[2/3] Sampling from full dataset...")
        pool_df = df
        print(f"  Full dataset pool size: {len(pool_df)} rows")

    effective_n = min(args.sample_size, len(pool_df))
    if effective_n < args.sample_size:
        print(f"  WARNING: Requested {args.sample_size} rows, only {len(pool_df)} available. Using {effective_n}.")

    sampled_idx = np.random.choice(len(pool_df), size=effective_n, replace=False)
    test_df = pool_df.iloc[sampled_idx].reset_index(drop=True)
    print(f"  Sampled rows: {len(test_df)}")

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
    print(f"  Sample source: {args.sample_source}")
    print(f"  Random seed: {effective_random_seed}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Sectors: {test_df['sector'].nunique()}")
    print(f"  Subcategories: {test_df['subcategory'].nunique()} (non-null: {test_df['subcategory'].notna().sum()})")
    print(f"  Types: {test_df['type'].nunique()} (non-null: {test_df['type'].notna().sum()})")
    if "tag" in test_df.columns:
        print(f"  Tags: {test_df['tag'].nunique()} (non-null: {test_df['tag'].notna().sum()})")

    print("\nNext steps:")
    print(f"  1. Run predictions:")
    print(f"     python predict_supervised.py --model_dir ./artifacts/<run_id> --input_xlsx {input_file} --output_xlsx ./data/test/test_predictions.xlsx")
    print(f"  2. Validate results:")
    print(f"     python validate_predictions.py --predictions_xlsx ./data/test/test_predictions.xlsx --truth_xlsx {truth_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
