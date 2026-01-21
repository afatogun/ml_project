#!/usr/bin/env python3
"""
Full test suite: Train, predict, and validate in one run.

Usage:
    python run_test_suite.py --train_xlsx ./data/raw/2025_classification_training_set.xlsx
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"\n  ERROR: {description} failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full test suite")
    parser.add_argument(
        "--train_xlsx",
        default="./data/raw/2025_classification_training_set.xlsx",
        help="Path to training XLSX file",
    )
    parser.add_argument(
        "--embedding_model",
        default="text-embedding-3-large",
        help="OpenAI embedding model name",
    )
    parser.add_argument(
        "--classifier",
        default="lightgbm",
        choices=["lightgbm", "random_forest", "logistic"],
        help="Classifier type",
    )
    parser.add_argument(
        "--subcategory_threshold",
        type=float,
        default=0.40,
        help="Confidence threshold for subcategory",
    )
    parser.add_argument(
        "--type_threshold",
        type=float,
        default=0.40,
        help="Confidence threshold for type",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.10,
        help="Fraction of data for test set",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training (use existing model)",
    )
    parser.add_argument(
        "--model_dir",
        help="Model directory to use (required if --skip_train)",
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  FULL TEST SUITE")
    print("="*60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Training data: {args.train_xlsx}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Classifier: {args.classifier}")
    print(f"  Thresholds: subcategory={args.subcategory_threshold}, type={args.type_threshold}")

    model_dir = args.model_dir

    # Step 1: Train
    if not args.skip_train:
        success = run_command([
            "python", "train.py",
            "--train_xlsx", args.train_xlsx,
            "--embedding_model", args.embedding_model,
            "--classifier", args.classifier,
            "--subcategory_threshold", str(args.subcategory_threshold),
            "--type_threshold", str(args.type_threshold),
            "--test_size", str(args.test_size),
        ], "STEP 1/4: Training models")

        if not success:
            return 1

        # Find the latest model directory
        artifacts_dir = "./artifacts"
        runs = sorted([d for d in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, d))])
        if not runs:
            print("ERROR: No model artifacts found")
            return 1
        model_dir = os.path.join(artifacts_dir, runs[-1])
        print(f"\n  Model saved to: {model_dir}")
    else:
        if not model_dir:
            print("ERROR: --model_dir required when using --skip_train")
            return 1
        print(f"\n  Skipping training, using model: {model_dir}")

    # Step 2: Create test input
    success = run_command([
        "python", "create_test_input.py",
        "--train_xlsx", args.train_xlsx,
        "--test_size", str(args.test_size),
        "--output_dir", "./data/test",
    ], "STEP 2/4: Creating test input files")

    if not success:
        return 1

    # Step 3: Run predictions
    success = run_command([
        "python", "predict_xlsx.py",
        "--model_dir", model_dir,
        "--input_xlsx", "./data/test/test_input.xlsx",
        "--output_xlsx", "./data/test/test_predictions.xlsx",
        "--subcategory_threshold", str(args.subcategory_threshold),
        "--type_threshold", str(args.type_threshold),
    ], "STEP 3/4: Running predictions on test set")

    if not success:
        return 1

    # Step 4: Validate predictions
    success = run_command([
        "python", "validate_predictions.py",
        "--predictions_xlsx", "./data/test/test_predictions.xlsx",
        "--truth_xlsx", "./data/test/test_ground_truth.xlsx",
        "--output_csv", "./data/test/comparison.csv",
    ], "STEP 4/4: Validating predictions")

    if not success:
        return 1

    # Summary
    print("\n" + "="*60)
    print("  TEST SUITE COMPLETE")
    print("="*60)
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {model_dir}")
    print(f"  Predictions: ./data/test/test_predictions.xlsx")
    print(f"  Comparison: ./data/test/comparison.csv")
    print(f"  Reports: ./reports/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
