"""Model training utilities."""

import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from .preprocess import get_eligible_subcategory_mask, get_eligible_type_mask


SEED = 42


def stratified_split(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    test_size: float = 0.10,
) -> Dict[str, Any]:
    """
    Stratified train/test split by Sector.

    Uses 90% for training, 10% for testing to maximize training data
    while keeping a holdout set for production accuracy validation.

    Returns dict with keys: train, test
    Each contains: df, embeddings, indices
    """
    np.random.seed(SEED)

    train_idx, test_idx = stratified_split_indices(df, test_size=test_size)

    return {
        "train": {
            "df": df.iloc[train_idx].reset_index(drop=True),
            "embeddings": embeddings[train_idx],
            "indices": train_idx,
        },
        "test": {
            "df": df.iloc[test_idx].reset_index(drop=True),
            "embeddings": embeddings[test_idx],
            "indices": test_idx,
        },
        "seed": SEED,
        "test_size": test_size,
    }


def stratified_split_indices(
    df: pd.DataFrame,
    test_size: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return reproducible stratified train/test indices by sector."""
    np.random.seed(SEED)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=df["sector"],
        random_state=SEED,
    )
    return train_idx, test_idx


def save_split_manifest(
    splits: Dict[str, Any],
    reports_dir: str,
) -> str:
    """Persist split membership for reproducible benchmarking."""
    os.makedirs(reports_dir, exist_ok=True)

    rows = []
    for split_name in ["train", "test"]:
        split_df = splits[split_name]["df"]
        split_indices = splits[split_name]["indices"]
        for split_row, original_index in zip(split_df.itertuples(index=False), split_indices):
            rows.append({
                "split": split_name,
                "original_index": int(original_index),
                "id": getattr(split_row, "id", None),
                "sector": getattr(split_row, "sector", None),
                "description": getattr(split_row, "description", None),
            })

    manifest_df = pd.DataFrame(rows)
    output_path = os.path.join(reports_dir, "split_manifest.csv")
    manifest_df.to_csv(output_path, index=False)

    metadata_path = os.path.join(reports_dir, "split_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": splits.get("seed", SEED),
                "test_size": splits.get("test_size"),
                "n_train": int(len(splits["train"]["df"])),
                "n_test": int(len(splits["test"]["df"])),
            },
            handle,
            indent=2,
        )

    return output_path


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: Literal["logistic", "lightgbm", "random_forest"] = "lightgbm",
) -> Tuple[Any, LabelEncoder]:
    """
    Train a classifier with label encoding.

    Args:
        X: Feature matrix (embeddings)
        y: Labels
        classifier_type: "logistic", "lightgbm", or "random_forest"

    Returns: (model, label_encoder)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    if classifier_type == "lightgbm":
        # LightGBM - excellent for high-dimensional data
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=15,
            num_leaves=64,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )
    elif classifier_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    else:  # logistic
        model = LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
        )

    model.fit(X, y_encoded)
    return model, le


def train_all_classifiers(
    splits: Dict[str, Any],
    classifier_type: str = "lightgbm",
) -> Dict[str, Any]:
    """
    Train all 4 classifiers using training split.

    Returns dict with models and encoders.
    """
    train_df = splits["train"]["df"]
    train_emb = splits["train"]["embeddings"]

    # 1. Sector classifier (all rows)
    print(f"    Training sector classifier ({classifier_type})...")
    sector_model, sector_le = train_classifier(
        train_emb,
        train_df["sector"].values,
        classifier_type=classifier_type,
    )

    # 2. Subcategory classifier (eligible rows only)
    subcat_mask = get_eligible_subcategory_mask(train_df)
    subcat_model, subcat_le = None, None
    if subcat_mask.sum() > 0:
        print(f"    Training subcategory classifier ({classifier_type})...")
        subcat_model, subcat_le = train_classifier(
            train_emb[subcat_mask],
            train_df.loc[subcat_mask, "subcategory"].values,
            classifier_type=classifier_type,
        )

    # 3. Type classifier (eligible rows only)
    type_mask = get_eligible_type_mask(train_df)
    type_model, type_le = None, None
    if type_mask.sum() > 0:
        print(f"    Training type classifier ({classifier_type})...")
        type_model, type_le = train_classifier(
            train_emb[type_mask],
            train_df.loc[type_mask, "type"].values,
            classifier_type=classifier_type,
        )

    # 4. Tag classifier (binary: "garage" vs "none" on ALL rows)
    #    Training only on garage-labelled rows would create a single-class model
    #    that predicts "garage" for everything. Instead train on all rows.
    tag_model, tag_le = None, None
    if train_df["tag"].notna().sum() > 0:
        print(f"    Training tag classifier ({classifier_type})...")
        tag_labels = train_df["tag"].fillna("none").values
        tag_model, tag_le = train_classifier(
            train_emb,
            tag_labels,
            classifier_type=classifier_type,
        )

    return {
        "sector": {"model": sector_model, "encoder": sector_le},
        "subcategory": {"model": subcat_model, "encoder": subcat_le},
        "type": {"model": type_model, "encoder": type_le},
        "tag": {"model": tag_model, "encoder": tag_le},
    }


def save_artifacts(
    classifiers: Dict[str, Any],
    output_dir: str,
    embedding_model: str,
    data_hash: str,
    metrics: Dict[str, Any],
    thresholds: Dict[str, float],
    classifier_type: str = "lightgbm",
        test_indices: Optional[np.ndarray] = None,
) -> str:
    """
    Save all artifacts to output directory.


        Args:
            test_indices: Optional test indices for reproducible validation.

        Returns the run_id.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = os.path.join(output_dir, run_id)
    os.makedirs(artifact_dir, exist_ok=True)

    # Save models and encoders
    for name, obj in classifiers.items():
        if obj["model"] is not None:
            joblib.dump(obj["model"], os.path.join(artifact_dir, f"{name}_model.joblib"))
            joblib.dump(obj["encoder"], os.path.join(artifact_dir, f"{name}_encoder.joblib"))

    # Save test indices if provided (for reproducible validation)
    if test_indices is not None:
        np.save(os.path.join(artifact_dir, "split_indices.npy"), test_indices)

    # Save metadata
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "embedding_model": embedding_model,
        "classifier_type": classifier_type,
        "data_hash": data_hash,
        "thresholds": thresholds,
        "metrics": metrics,
        "classifiers": {
            name: obj["model"] is not None
            for name, obj in classifiers.items()
        },
        "test_indices_count": len(test_indices) if test_indices is not None else None,
        "test_indices_file": "split_indices.npy" if test_indices is not None else None,
    }

    with open(os.path.join(artifact_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return run_id
