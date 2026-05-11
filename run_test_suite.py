#!/usr/bin/env python3
"""
Full test suite: Train, predict, and validate in one run.

Usage:
    python run_test_suite.py --train_xlsx ./data/raw/2025_classification_training_set.xlsx
"""

import argparse
import json
import os
import sys
import subprocess
import secrets
from datetime import datetime

from src.preprocess import compute_data_hash, load_training_data


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


def resolve_train_path(train_xlsx: str) -> str:
    """Resolve training file path with sensible workspace defaults."""
    if os.path.exists(train_xlsx):
        return train_xlsx

    basename = os.path.basename(train_xlsx)
    candidates = [
        os.path.join(".", "data", "raw", basename),
        os.path.join(".", basename),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find training file '{train_xlsx}'. Tried: {', '.join(candidates)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run supervised-only test suite")
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
        "--test_size",
        type=float,
        default=0.10,
        help="Fraction of data for test set",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of projects to sample for prediction/validation (default: 1000)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sample reproducibility (default: 42)",
    )
    parser.add_argument(
        "--random_mode",
        action="store_true",
        help="Generate a fresh random seed for sampling on each run",
    )
    parser.add_argument(
        "--sample_source",
        choices=["full", "stratified_test"],
        default="full",
        help="Sampling pool used by create_test_input.py (default: full)",
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
    parser.add_argument(
        "--supervised_candidate",
        default="mlp_embedding_rule",
        choices=[
            "embedding_lgbm", "embedding_rule_lgbm", "mlp_embedding_rule",
            "stacked_mlp_lgbm", "candidate_d_tfidf_svd_rule",
        ],
        help="Supervised candidate to use (default: mlp_embedding_rule)",
    )

    args = parser.parse_args()

    effective_random_seed = secrets.randbelow(2**31 - 1) if args.random_mode else args.random_seed

    resolved_train_xlsx = resolve_train_path(args.train_xlsx)
    expected_data_hash = compute_data_hash(load_training_data(resolved_train_xlsx))

    print("\n" + "="*60)
    print("  FULL TEST SUITE")
    print("="*60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Mode: Supervised-only")
    print(f"  Training data (resolved): {resolved_train_xlsx}")
    print(f"  Sample source: {args.sample_source}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Random mode: {args.random_mode}")
    print(f"  Random seed: {effective_random_seed}")
    print(f"  Data hash (from train_xlsx): {expected_data_hash}")
    print(f"  Training data: {resolved_train_xlsx}")
    print(f"  Supervised candidate: {args.supervised_candidate}")

    model_dir = args.model_dir
    model_run_id = None
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Train supervised model unless skipped
    if not args.skip_train:
        train_cmd = [
            "python", "train_supervised_sector.py",
            "--train_xlsx", resolved_train_xlsx,
            "--embedding_model", args.embedding_model,
            "--candidate", args.supervised_candidate,
        ]
        success = run_command(train_cmd, "STEP 1/4: Training supervised sector model")
        if not success:
            return 1
        artifacts_dir = "./artifacts"
        runs = sorted([
            d for d in os.listdir(artifacts_dir)
            if os.path.isdir(os.path.join(artifacts_dir, d)) and d.endswith("_supervised")
        ])
        if not runs:
            print("ERROR: No supervised model artifacts found")
            return 1
        model_dir = os.path.join(artifacts_dir, runs[-1])
        model_run_id = os.path.basename(model_dir.rstrip("/"))
        print(f"\n  Supervised model saved to: {model_dir}")
    else:
        if not model_dir:
            print("ERROR: --model_dir required when using --skip_train")
            return 1
        model_run_id = os.path.basename(model_dir.rstrip("/"))
        print(f"\n  Skipping training, using model: {model_dir}")

    # Validate supervised artifact compatibility against current train_xlsx.
    supervised_meta_path = os.path.join(model_dir, "metadata.json")
    with open(supervised_meta_path, "r", encoding="utf-8") as handle:
        supervised_meta = json.load(handle)
    supervised_hash = supervised_meta.get("data_hash")
    if supervised_hash is None:
        print("\nWARNING: Supervised artifact has no data_hash metadata; cannot verify training-data compatibility.")
        print("         Re-train once with current code to stamp hash into metadata.")
    elif supervised_hash != expected_data_hash:
        print("\nERROR: Supervised artifact data hash does not match --train_xlsx")
        print(f"  Artifact hash: {supervised_hash}")
        print(f"  Current  hash: {expected_data_hash}")
        print("  Use a matching supervised artifact or retrain with this training file.")
        return 1

    test_dir = os.path.join("./data/test", run_id)
    input_xlsx = os.path.join(test_dir, "test_input.xlsx")
    truth_xlsx = os.path.join(test_dir, "test_ground_truth.xlsx")
    predictions_xlsx = os.path.join(test_dir, "test_predictions.xlsx")
    comparison_csv = os.path.join(test_dir, "comparison.csv")

    # Determine split indices path (for reproducible test-set sampling)
    split_indices_path = None
    if args.sample_source == "stratified_test":
        split_indices_path = os.path.join(model_dir, "split_indices.npy")
        if not os.path.exists(split_indices_path):
            print(f"\nWARNING: stratified_test selected, but split_indices.npy not found at {split_indices_path}")
            print("         This means the model was trained without saved test indices.")
            print("         Consider retraining the model to ensure reproducible test/train separation.")
            split_indices_path = None

    # Step 2: Create test input
    test_input_cmd = [
        "python", "create_test_input.py",
        "--train_xlsx", resolved_train_xlsx,
        "--test_size", str(args.test_size),
        "--sample_size", str(args.sample_size),
        "--random_seed", str(effective_random_seed),
        "--sample_source", args.sample_source,
        "--output_dir", test_dir,
    ]

    # Add split indices path if available (for reproducible stratified sampling)
    if split_indices_path:
        test_input_cmd.extend(["--split_indices_path", split_indices_path])

    success = run_command(test_input_cmd, "STEP 2/4: Creating test input files")

    if not success:
        return 1

    # Step 3: Run predictions
    predict_cmd = [
        "python", "predict_supervised.py",
        "--model_dir", model_dir,
        "--input_xlsx", input_xlsx,
        "--output_xlsx", predictions_xlsx,
    ]
    success = run_command(predict_cmd, "STEP 3/4: Running supervised predictions on test set")

    if not success:
        return 1

    # Step 4: Validate predictions
    validate_cmd = [
        "python", "validate_ai.py",
        "--predictions", predictions_xlsx,
        "--truth", truth_xlsx,
        "--output-dir", "./reports",
    ]
    if run_id:
        validate_cmd.extend(["--run-id", run_id])
    success = run_command(validate_cmd, "STEP 4/4: Validating supervised predictions")

    if not success:
        return 1

    # Summary
    print("\n" + "="*60)
    print("  TEST SUITE COMPLETE")
    print("="*60)
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: Supervised-first ({args.supervised_candidate})")
    print(f"  Model: {model_dir}")
    if model_run_id:
        print(f"  Source model run: {model_run_id}")
    print(f"  Predictions: {predictions_xlsx}")
    print(f"  Comparison: ./reports/{run_id}/comparison_ai.csv")
    print(f"  Reports: ./reports/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
