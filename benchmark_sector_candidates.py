#!/usr/bin/env python3
"""Benchmark first-wave supervised sector candidates on a stable split."""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

from src.openai_embeddings import batch_embed
from src.preprocess import compute_data_hash, load_training_data
from src.supervised_candidates import benchmark_sector_candidates
from src.train_models import SEED, save_split_manifest, stratified_split_indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark supervised sector candidates")
    parser.add_argument("--train_xlsx", required=True, help="Path to labelled training data")
    parser.add_argument(
        "--embedding_model",
        default="text-embedding-3-large",
        help="Embedding model for embedding-based candidate",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="Embedding batch size")
    parser.add_argument("--test_size", type=float, default=0.10, help="Holdout fraction")
    parser.add_argument("--reports_dir", default="./reports", help="Directory to save benchmark reports")
    args = parser.parse_args()

    load_dotenv()
    np.random.seed(SEED)

    print("=" * 60)
    print("SUPERVISED SECTOR CANDIDATE BENCHMARK")
    print("=" * 60)

    print(f"\n[1/4] Loading data from: {args.train_xlsx}")
    df = load_training_data(args.train_xlsx)
    print(f"  Loaded {len(df)} rows")
    print(f"  Data hash: {compute_data_hash(df)}")

    print(f"\n[2/4] Computing embeddings with {args.embedding_model}...")
    embeddings = batch_embed(
        df["description"].tolist(),
        model=args.embedding_model,
        batch_size=args.batch_size,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    print(f"\n[3/4] Creating stable split ({int((1-args.test_size)*100)}/{int(args.test_size*100)})...")
    train_idx, test_idx = stratified_split_indices(df, test_size=args.test_size)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_embeddings = embeddings[train_idx]
    test_embeddings = embeddings[test_idx]

    report_run_dir = os.path.join(args.reports_dir, datetime.now().strftime("%Y%m%d_%H%M%S_candidates"))
    os.makedirs(report_run_dir, exist_ok=True)
    save_split_manifest(
        {
            "train": {"df": train_df, "indices": train_idx},
            "test": {"df": test_df, "indices": test_idx},
            "seed": SEED,
            "test_size": args.test_size,
        },
        report_run_dir,
    )
    print(f"  Train rows: {len(train_df)}")
    print(f"  Test rows:  {len(test_df)}")

    print("\n[4/4] Training and evaluating candidates...")
    results = benchmark_sector_candidates(
        train_df=train_df,
        test_df=test_df,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        reports_dir=report_run_dir,
    )

    for candidate_name, metrics in results.items():
        print(f"  {candidate_name}:")
        print(f"    Accuracy: {metrics['accuracy']}")
        print(f"    Macro-F1: {metrics['macro_f1']}")
        print(f"    Samples:  {metrics['n_samples']}")

    print(f"\nReports saved to: {report_run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())