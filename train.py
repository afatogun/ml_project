#!/usr/bin/env python3
"""
Training script for project classification models.

Usage:
    python train.py --train_xlsx path/to/training.xlsx
"""

import argparse
import os
import sys

import numpy as np
from dotenv import load_dotenv

from src.preprocess import load_training_data, compute_data_hash
from src.openai_embeddings import batch_embed
from src.train_models import stratified_split, train_all_classifiers, save_artifacts, SEED
from src.metrics import evaluate_all, print_metrics


def main():
    parser = argparse.ArgumentParser(description="Train classification models")
    parser.add_argument(
        "--train_xlsx",
        required=True,
        help="Path to training XLSX file",
    )
    parser.add_argument(
        "--embedding_model",
        default="text-embedding-3-large",
        help="OpenAI embedding model name (text-embedding-3-large recommended for higher accuracy)",
    )
    parser.add_argument(
        "--classifier",
        default="lightgbm",
        choices=["lightgbm", "random_forest", "logistic"],
        help="Classifier type: lightgbm (best), random_forest, or logistic",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for embedding API calls",
    )
    parser.add_argument(
        "--output_dir",
        default="./artifacts",
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--reports_dir",
        default="./reports",
        help="Directory to save evaluation reports",
    )
    parser.add_argument(
        "--subcategory_threshold",
        type=float,
        default=0.60,
        help="Confidence threshold for subcategory predictions",
    )
    parser.add_argument(
        "--type_threshold",
        type=float,
        default=0.60,
        help="Confidence threshold for type predictions",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.10,
        help="Fraction of data to hold out for testing (default: 0.10 = 10%%)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Set seeds
    np.random.seed(SEED)

    print("=" * 60)
    print("PROJECT CLASSIFICATION TRAINING")
    print("=" * 60)

    # 1. Load and preprocess data
    print(f"\n[1/5] Loading training data from: {args.train_xlsx}")
    df = load_training_data(args.train_xlsx)
    data_hash = compute_data_hash(df)
    print(f"  Loaded {len(df)} rows")
    print(f"  Data hash: {data_hash}")
    print(f"  Sectors: {df['sector'].nunique()}")
    print(f"  Subcategories: {df['subcategory'].nunique()} (eligible rows: {df['subcategory'].notna().sum()})")
    print(f"  Types: {df['type'].nunique()} (eligible rows: {df['type'].notna().sum()})")

    # 2. Compute embeddings
    print(f"\n[2/5] Computing embeddings with {args.embedding_model}...")
    descriptions = df["description"].tolist()
    embeddings = batch_embed(
        descriptions,
        model=args.embedding_model,
        batch_size=args.batch_size,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # 3. Split data
    print(f"\n[3/5] Creating stratified train/test split ({int((1-args.test_size)*100)}/{int(args.test_size*100)})...")
    splits = stratified_split(df, embeddings, test_size=args.test_size)
    print(f"  Train: {len(splits['train']['df'])} rows ({int((1-args.test_size)*100)}%)")
    print(f"  Test:  {len(splits['test']['df'])} rows ({int(args.test_size*100)}% holdout for accuracy validation)")

    # 4. Train classifiers
    print(f"\n[4/5] Training classifiers with {args.classifier}...")
    classifiers = train_all_classifiers(splits, classifier_type=args.classifier)
    for name, obj in classifiers.items():
        status = "trained" if obj["model"] is not None else "skipped (no eligible data)"
        print(f"  {name}: {status}")

    # 5. Evaluate
    print("\n[5/5] Evaluating models...")
    os.makedirs(args.reports_dir, exist_ok=True)
    metrics = evaluate_all(classifiers, splits, args.reports_dir)
    print_metrics(metrics)

    # Save artifacts
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    thresholds = {
        "subcategory": args.subcategory_threshold,
        "type": args.type_threshold,
    }

    run_id = save_artifacts(
        classifiers=classifiers,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        data_hash=data_hash,
        metrics=metrics,
        thresholds=thresholds,
        classifier_type=args.classifier,
    )

    print(f"\nArtifacts saved to: {os.path.join(args.output_dir, run_id)}")
    print(f"Confusion matrices saved to: {args.reports_dir}")
    print("\nTraining complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
