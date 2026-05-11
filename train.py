#!/usr/bin/env python3
"""
Training script for project classification models.

Usage:
    python train.py --train_xlsx path/to/training.xlsx
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

from src.preprocess import load_training_data, compute_data_hash, apply_translation
from src.openai_embeddings import batch_embed
from src.train_models import stratified_split, train_all_classifiers, save_artifacts, save_split_manifest, SEED
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
        "--tag_threshold",
        type=float,
        default=0.60,
        help="Confidence threshold for tag predictions",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.10,
        help="Fraction of data to hold out for testing (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enable Irish-to-English translation before training",
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
    print(f"\n[1/6] Loading training data from: {args.train_xlsx}")
    df = load_training_data(args.train_xlsx)
    print(f"  Loaded {len(df)} rows")

    # 1.5. Apply translation if enabled
    if args.translate:
        print(f"\n[1.5/6] Applying Irish-to-English translation...")
        df = apply_translation(df, text_column="description")
        translated_count = df["translation_applied"].sum()
        print(f"  Translated {translated_count} rows")

    data_hash = compute_data_hash(df)
    print(f"  Data hash: {data_hash}")
    print(f"  Sectors: {df['sector'].nunique()}")
    print(f"  Subcategories: {df['subcategory'].nunique()} (eligible rows: {df['subcategory'].notna().sum()})")
    print(f"  Types: {df['type'].nunique()} (eligible rows: {df['type'].notna().sum()})")
    print(f"  Tags: {df['tag'].nunique()} (eligible rows: {df['tag'].notna().sum()})")

    # 2. Compute embeddings
    print(f"\n[2/6] Computing embeddings with {args.embedding_model}...")
    descriptions = df["description"].tolist()
    embeddings = batch_embed(
        descriptions,
        model=args.embedding_model,
        batch_size=args.batch_size,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # 3. Split data
    print(f"\n[3/6] Creating stratified train/test split ({int((1-args.test_size)*100)}/{int(args.test_size*100)})...")
    splits = stratified_split(df, embeddings, test_size=args.test_size)
    print(f"  Train: {len(splits['train']['df'])} rows ({int((1-args.test_size)*100)}%)")
    print(f"  Test:  {len(splits['test']['df'])} rows ({int(args.test_size*100)}% holdout for accuracy validation)")

    report_run_dir = os.path.join(args.reports_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    split_manifest_path = save_split_manifest(splits, report_run_dir)
    print(f"  Split manifest: {split_manifest_path}")

    # 4. Train classifiers
    print(f"\n[4/6] Training classifiers with {args.classifier}...")
    classifiers = train_all_classifiers(splits, classifier_type=args.classifier)
    for name, obj in classifiers.items():
        status = "trained" if obj["model"] is not None else "skipped (no eligible data)"
        print(f"  {name}: {status}")

    # 5. Evaluate
    print("\n[5/6] Evaluating models...")
    os.makedirs(report_run_dir, exist_ok=True)
    metrics = evaluate_all(classifiers, splits, report_run_dir)
    print_metrics(metrics)

    # 6. Save artifacts
    print("\n[6/6] Saving artifacts...")
    print("=" * 60)

    thresholds = {
        "subcategory": args.subcategory_threshold,
        "type": args.type_threshold,
        "tag": args.tag_threshold,
    }

    run_id = save_artifacts(
        classifiers=classifiers,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        data_hash=data_hash,
        metrics=metrics,
        thresholds=thresholds,
        classifier_type=args.classifier,
        test_indices=splits["test"]["indices"],
    )

    print(f"\nArtifacts saved to: {os.path.join(args.output_dir, run_id)}")
    print(f"Reports saved to: {report_run_dir}")
    print("\nTraining complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
