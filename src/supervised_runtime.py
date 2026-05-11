"""Runtime utilities for supervised-first sector models."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .supervised_candidates import (
    build_rule_feature_frame,
    train_embedding_lgbm_sector_model,
    train_embedding_rule_lgbm_sector_model,
    train_mlp_sector_model,
    train_stacked_sector_model,
    train_tfidf_logistic_sector_model,
    train_tfidf_reduced_rule_hybrid_sector_model,
)


SUPPORTED_CANDIDATES = {
    "embedding_lgbm",
    "embedding_rule_lgbm",
    "mlp_embedding_rule",
    "stacked_mlp_lgbm",
    "candidate_d_tfidf_svd_rule",
    "candidate_e_probability_ensemble",
}


def _align_probabilities(source_encoder, target_encoder, probabilities: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(target_encoder.classes_)), dtype=np.float32)
    target_lookup = {label: idx for idx, label in enumerate(target_encoder.classes_)}
    for source_idx, label in enumerate(source_encoder.classes_):
        aligned[:, target_lookup[label]] = probabilities[:, source_idx]
    return aligned


def train_supervised_sector_bundle(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    candidate_name: str,
    embedding_model_name: str,
) -> Dict[str, Any]:
    if candidate_name not in SUPPORTED_CANDIDATES:
        raise ValueError(f"Unsupported candidate: {candidate_name}")

    texts = df["description"].fillna("").astype(str).tolist()
    y = df["sector"].values

    bundle: Dict[str, Any] = {
        "candidate_name": candidate_name,
        "embedding_model": embedding_model_name,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_rows": int(len(df)),
    }

    if candidate_name == "embedding_lgbm":
        model, encoder = train_embedding_lgbm_sector_model(embeddings, y)
        bundle.update({"encoder": encoder, "model": model})
        return bundle

    if candidate_name == "embedding_rule_lgbm":
        model, encoder = train_embedding_rule_lgbm_sector_model(embeddings, texts, y)
        bundle.update({"encoder": encoder, "model": model})
        return bundle

    if candidate_name == "mlp_embedding_rule":
        model, encoder, scaler = train_mlp_sector_model(embeddings, texts, y)
        bundle.update({"encoder": encoder, "model": model, "scaler": scaler})
        return bundle

    if candidate_name == "stacked_mlp_lgbm":
        mlp_model, lgbm_model, meta_model, encoder, scaler = train_stacked_sector_model(embeddings, texts, y)
        bundle.update({
            "encoder": encoder,
            "mlp_model": mlp_model,
            "lgbm_model": lgbm_model,
            "meta_model": meta_model,
            "scaler": scaler,
        })
        return bundle

    if candidate_name == "candidate_d_tfidf_svd_rule":
        model, encoder = train_tfidf_reduced_rule_hybrid_sector_model(texts, embeddings, y)
        bundle.update({"encoder": encoder, "model": model})
        return bundle

    # Candidate E ensemble
    embedding_model, embedding_encoder = train_embedding_lgbm_sector_model(embeddings, y)
    tfidf_model, tfidf_encoder = train_tfidf_logistic_sector_model(texts, y)
    candidate_d_model, candidate_d_encoder = train_tfidf_reduced_rule_hybrid_sector_model(texts, embeddings, y)
    bundle.update(
        {
            "encoder": embedding_encoder,
            "embedding_model_obj": embedding_model,
            "tfidf_model_obj": tfidf_model,
            "tfidf_encoder": tfidf_encoder,
            "candidate_d_model_obj": candidate_d_model,
            "candidate_d_encoder": candidate_d_encoder,
        }
    )
    return bundle


def predict_supervised_sector(
    bundle: Dict[str, Any],
    descriptions: List[str],
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[List[float]]]:
    candidate_name = bundle["candidate_name"]
    if candidate_name not in SUPPORTED_CANDIDATES:
        raise ValueError(f"Unsupported candidate: {candidate_name}")

    texts = [str(d or "") for d in descriptions]
    encoder = bundle["encoder"]

    if candidate_name == "embedding_lgbm":
        probabilities = bundle["model"].predict_proba(embeddings)
    elif candidate_name == "embedding_rule_lgbm":
        rule_features = build_rule_feature_frame(texts).to_numpy(dtype=np.float32)
        matrix = np.hstack([embeddings, rule_features])
        probabilities = bundle["model"].predict_proba(matrix)
    elif candidate_name == "mlp_embedding_rule":
        rule_features = build_rule_feature_frame(texts).to_numpy(dtype=np.float32)
        matrix = bundle["scaler"].transform(np.hstack([embeddings, rule_features]))
        probabilities = bundle["model"].predict_proba(matrix)
    elif candidate_name == "stacked_mlp_lgbm":
        rule_features = build_rule_feature_frame(texts).to_numpy(dtype=np.float32)
        n_emb = embeddings.shape[1]
        n_rule = rule_features.shape[1]
        mlp_matrix = bundle["scaler"].transform(np.hstack([embeddings, rule_features]))
        lgbm_df = pd.DataFrame(
            np.hstack([embeddings, rule_features]),
            columns=[f"f{i}" for i in range(n_emb + n_rule)],
        )
        mlp_probs = bundle["mlp_model"].predict_proba(mlp_matrix)
        lgbm_probs = bundle["lgbm_model"].predict_proba(lgbm_df)
        probabilities = bundle["meta_model"].predict_proba(np.hstack([mlp_probs, lgbm_probs]))
    elif candidate_name == "candidate_d_tfidf_svd_rule":
        rule_features = build_rule_feature_frame(texts).to_numpy(dtype=np.float32)
        probabilities = bundle["model"].predict_proba(texts, embeddings, rule_features)
    else:
        rule_features = build_rule_feature_frame(texts).to_numpy(dtype=np.float32)
        embedding_probabilities = bundle["embedding_model_obj"].predict_proba(embeddings)
        tfidf_probabilities = bundle["tfidf_model_obj"].predict_proba(texts)
        tfidf_probabilities = _align_probabilities(bundle["tfidf_encoder"], encoder, tfidf_probabilities)
        candidate_d_probabilities = bundle["candidate_d_model_obj"].predict_proba(texts, embeddings, rule_features)
        candidate_d_probabilities = _align_probabilities(bundle["candidate_d_encoder"], encoder, candidate_d_probabilities)
        probabilities = (embedding_probabilities + tfidf_probabilities + candidate_d_probabilities) / 3.0

    top_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
    top_labels = [[str(encoder.classes_[idx]) for idx in row] for row in top_indices]
    top_probs = [[round(float(probabilities[row_i, idx]), 6) for idx in row] for row_i, row in enumerate(top_indices)]
    pred_indices = top_indices[:, 0]
    pred_labels = encoder.inverse_transform(pred_indices)
    pred_conf = probabilities[np.arange(len(probabilities)), pred_indices]
    return pred_labels, pred_conf, top_labels, top_probs


def save_supervised_bundle(bundle: Dict[str, Any], output_dir: str) -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_supervised")
    artifact_dir = os.path.join(output_dir, run_id)
    os.makedirs(artifact_dir, exist_ok=True)

    joblib.dump(bundle, os.path.join(artifact_dir, "sector_supervised_bundle.joblib"))

    # Save test indices if present (for reproducible validation)
    if "test_indices" in bundle:
        test_indices = np.array(bundle["test_indices"])
        np.save(os.path.join(artifact_dir, "split_indices.npy"), test_indices)

    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "candidate_name": bundle["candidate_name"],
                "embedding_model": bundle["embedding_model"],
                "n_rows": bundle["n_rows"],
                "model_type": "supervised_sector_bundle",
                "data_hash": bundle.get("data_hash"),
                "train_xlsx": bundle.get("train_xlsx"),
                "test_indices_count": bundle.get("test_indices") and len(bundle["test_indices"]) or None,
                "test_indices_file": "split_indices.npy" if "test_indices" in bundle else None,
            },
            handle,
            indent=2,
        )
    return artifact_dir


def load_supervised_bundle(model_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metadata_path = os.path.join(model_dir, "metadata.json")
    bundle_path = os.path.join(model_dir, "sector_supervised_bundle.joblib")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    bundle = joblib.load(bundle_path)
    return metadata, bundle
