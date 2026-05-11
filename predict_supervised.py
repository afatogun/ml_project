#!/usr/bin/env python3
"""Run supervised-first sector predictions from a saved bundle."""

import argparse
import json
import sys

import pandas as pd
from dotenv import load_dotenv

from src.openai_embeddings import batch_embed
from src.preprocess import load_inference_data
from src.supervised_runtime import load_supervised_bundle, predict_supervised_sector


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch predict sectors with supervised-first model")
    parser.add_argument("--model_dir", required=True, help="Path to supervised bundle artifact directory")
    parser.add_argument("--input_xlsx", required=True, help="Input XLSX/CSV with id, description")
    parser.add_argument("--output_xlsx", required=True, help="Output XLSX path")
    parser.add_argument("--batch_size", type=int, default=100, help="Embedding batch size")
    args = parser.parse_args()

    load_dotenv()

    print("=" * 60)
    print("SUPERVISED-FIRST SECTOR PREDICTION")
    print("=" * 60)

    print(f"\n[1/4] Loading bundle from: {args.model_dir}")
    metadata, bundle = load_supervised_bundle(args.model_dir)
    print(f"  Candidate: {metadata['candidate_name']}")
    print(f"  Embedding model: {metadata['embedding_model']}")

    print(f"\n[2/4] Loading input: {args.input_xlsx}")
    df = load_inference_data(args.input_xlsx)
    print(f"  Rows: {len(df)}")

    print(f"\n[3/4] Computing embeddings: {metadata['embedding_model']}")
    descriptions = df["description"].fillna("").astype(str).tolist()
    embeddings = batch_embed(descriptions, model=metadata["embedding_model"], batch_size=args.batch_size)

    print("\n[4/4] Running supervised predictions")
    pred_sector, pred_conf, top3_labels, top3_probs = predict_supervised_sector(bundle, descriptions, embeddings)

    output = pd.DataFrame(
        {
            "id": df["id"],
            "description": descriptions,
            "pred_sector": pred_sector,
            "pred_sector_conf": [round(float(v), 6) for v in pred_conf],
            "top_3_predicted_sectors": [json.dumps(v) for v in top3_labels],
            "top_3_predicted_probs": [json.dumps(v) for v in top3_probs],
            "prediction_source": ["supervised"] * len(df),
            "model_version": [metadata["run_id"]] * len(df),
            "candidate_name": [metadata["candidate_name"]] * len(df),
        }
    )

    output["pred_tag"] = "none"

    output.to_excel(args.output_xlsx, index=False)
    print(f"  Saved predictions to: {args.output_xlsx}")
    print("\nPrediction complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
