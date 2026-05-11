#!/usr/bin/env python3
"""Train a supervised-first sector model bundle for production inference."""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from src.openai_embeddings import batch_embed
from src.preprocess import compute_data_hash, load_training_data
from src.supervised_runtime import save_supervised_bundle, train_supervised_sector_bundle
from src.train_models import stratified_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Train supervised-first sector model")
    parser.add_argument("--train_xlsx", required=True, help="Path to labeled training data")
    parser.add_argument(
        "--candidate",
        default="mlp_embedding_rule",
        choices=[
            "embedding_lgbm",
            "embedding_rule_lgbm",
            "mlp_embedding_rule",
            "stacked_mlp_lgbm",
            "candidate_d_tfidf_svd_rule",
            "candidate_e_probability_ensemble",
        ],
        help="Sector candidate strategy to train",
    )
    parser.add_argument("--embedding_model", default="text-embedding-3-large", help="Embedding model")
    parser.add_argument("--batch_size", type=int, default=100, help="Embedding batch size")
    parser.add_argument("--output_dir", default="./artifacts", help="Artifact output directory")
    args = parser.parse_args()

    load_dotenv()

    print("=" * 60)
    print("TRAIN SUPERVISED-FIRST SECTOR MODEL")
    print("=" * 60)

    print(f"\n[1/4] Loading training data: {args.train_xlsx}")
    df = load_training_data(args.train_xlsx)
    data_hash = compute_data_hash(df)
    print(f"  Rows: {len(df)}")
    print(f"  Data hash: {data_hash}")

    print(f"\n[2/4] Computing embeddings: {args.embedding_model}")
    embeddings = batch_embed(df["description"].tolist(), model=args.embedding_model, batch_size=args.batch_size)
    print(f"  Embedding shape: {embeddings.shape}")

    print(f"\n[3/4] Creating 90/10 train/test split (for reproducible validation)")
    splits = stratified_split(df, embeddings, test_size=0.10)
    train_df = splits["train"]["df"]
    train_embeddings = splits["train"]["embeddings"]
    test_indices = splits["test"]["indices"]
    print(f"  Training rows: {len(train_df)}")
    print(f"  Test indices count: {len(test_indices)}")

    print(f"\n[4/4] Training candidate: {args.candidate}")
    bundle = train_supervised_sector_bundle(
        df=train_df,
        embeddings=train_embeddings,
        candidate_name=args.candidate,
        embedding_model_name=args.embedding_model,
    )
    bundle["data_hash"] = data_hash
    bundle["train_xlsx"] = os.path.abspath(args.train_xlsx)
    bundle["test_indices"] = test_indices.tolist()  # Save test indices for validation reproducibility
    artifact_dir = save_supervised_bundle(bundle, args.output_dir)

    # Save test indices to metadata for easy reference
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["test_indices_count"] = len(test_indices)
    metadata["test_indices_file"] = "split_indices.npy"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved supervised bundle to: {artifact_dir}")
    print(f"  Saved {len(test_indices)} test indices for reproducible validation")
    print("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
