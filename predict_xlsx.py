#!/usr/bin/env python3
"""
Batch prediction script for project classification.

Usage:
    python predict_xlsx.py --model_dir ./artifacts/<run_id> --input_xlsx input.xlsx --output_xlsx output.xlsx
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from src.preprocess import load_inference_data
from src.openai_embeddings import batch_embed
from src.infer import load_artifacts, run_inference


def main():
    parser = argparse.ArgumentParser(description="Batch predict classifications from XLSX")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to model artifacts directory (e.g., ./artifacts/20240115_120000)",
    )
    parser.add_argument(
        "--input_xlsx",
        required=True,
        help="Path to input XLSX file with columns: id, description",
    )
    parser.add_argument(
        "--output_xlsx",
        required=True,
        help="Path to output XLSX file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for embedding API calls",
    )
    parser.add_argument(
        "--subcategory_threshold",
        type=float,
        default=None,
        help="Override subcategory confidence threshold (default: use model's threshold)",
    )
    parser.add_argument(
        "--type_threshold",
        type=float,
        default=None,
        help="Override type confidence threshold (default: use model's threshold)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    print("=" * 60)
    print("PROJECT CLASSIFICATION PREDICTION")
    print("=" * 60)

    # 1. Load artifacts
    print(f"\n[1/4] Loading model artifacts from: {args.model_dir}")
    artifacts = load_artifacts(args.model_dir)
    metadata = artifacts["metadata"]
    print(f"  Run ID: {metadata['run_id']}")
    print(f"  Embedding model: {metadata['embedding_model']}")
    print(f"  Data hash: {metadata['data_hash']}")

    # Determine thresholds
    subcategory_threshold = args.subcategory_threshold or metadata["thresholds"]["subcategory"]
    type_threshold = args.type_threshold or metadata["thresholds"]["type"]
    print(f"  Subcategory threshold: {subcategory_threshold}")
    print(f"  Type threshold: {type_threshold}")

    # 2. Load input data
    print(f"\n[2/4] Loading input data from: {args.input_xlsx}")
    df_input = load_inference_data(args.input_xlsx)
    print(f"  Loaded {len(df_input)} rows")

    # 3. Compute embeddings
    print(f"\n[3/4] Computing embeddings...")
    descriptions = df_input["description"].tolist()
    embeddings = batch_embed(
        descriptions,
        model=metadata["embedding_model"],
        batch_size=args.batch_size,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # 4. Run inference
    print(f"\n[4/4] Running inference...")
    df_predictions = run_inference(
        embeddings=embeddings,
        artifacts=artifacts,
        subcategory_threshold=subcategory_threshold,
        type_threshold=type_threshold,
    )

    # Combine input with predictions
    df_output = pd.concat([
        df_input[["id", "description"]],
        df_predictions,
    ], axis=1)

    # Add model version
    df_output["model_version"] = metadata["run_id"]

    # Reorder columns
    column_order = [
        "id", "description",
        "pred_sector", "pred_sector_conf",
        "pred_subcategory", "pred_subcategory_conf",
        "pred_type", "pred_type_conf",
        "notes", "model_version",
    ]
    df_output = df_output[column_order]

    # Save output
    print(f"\nSaving output to: {args.output_xlsx}")
    df_output.to_excel(args.output_xlsx, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total rows: {len(df_output)}")
    print(f"  Sector predictions: {df_output['pred_sector'].notna().sum()}")
    print(f"  Subcategory predictions: {df_output['pred_subcategory'].notna().sum()}")
    print(f"  Type predictions: {df_output['pred_type'].notna().sum()}")
    print(f"  Rows with notes: {(df_output['notes'] != '').sum()}")

    print("\nPrediction complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
