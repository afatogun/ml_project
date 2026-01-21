"""Evaluation metrics and reporting."""

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .preprocess import get_eligible_subcategory_mask, get_eligible_type_mask


def evaluate_classifier(
    model,
    encoder,
    X: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a classifier and return metrics.
    """
    if model is None or len(X) == 0:
        return {"accuracy": None, "macro_f1": None, "n_samples": 0}

    y_pred_encoded = model.predict(X)
    y_pred = encoder.inverse_transform(y_pred_encoded)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "n_samples": len(y_true),
    }


def compute_confusion_matrix(
    model,
    encoder,
    X: np.ndarray,
    y_true: np.ndarray,
) -> Optional[pd.DataFrame]:
    """
    Compute confusion matrix as a DataFrame.
    """
    if model is None or len(X) == 0:
        return None

    y_pred_encoded = model.predict(X)
    y_pred = encoder.inverse_transform(y_pred_encoded)

    labels = encoder.classes_
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return pd.DataFrame(cm, index=labels, columns=labels)


def evaluate_all(
    classifiers: Dict[str, Any],
    splits: Dict[str, Any],
    reports_dir: str = "./reports",
) -> Dict[str, Any]:
    """
    Evaluate all classifiers on test split.
    Save confusion matrices to reports_dir.
    """
    os.makedirs(reports_dir, exist_ok=True)

    results = {}

    for split_name in ["test"]:
        split_data = splits[split_name]
        df = split_data["df"]
        emb = split_data["embeddings"]

        results[split_name] = {}

        # Sector (all rows)
        sector_metrics = evaluate_classifier(
            classifiers["sector"]["model"],
            classifiers["sector"]["encoder"],
            emb,
            df["sector"].values,
        )
        results[split_name]["sector"] = sector_metrics

        cm_sector = compute_confusion_matrix(
            classifiers["sector"]["model"],
            classifiers["sector"]["encoder"],
            emb,
            df["sector"].values,
        )
        if cm_sector is not None:
            cm_sector.to_csv(os.path.join(reports_dir, f"confusion_sector_{split_name}.csv"))

        # Subcategory (eligible rows only)
        subcat_mask = get_eligible_subcategory_mask(df)
        if subcat_mask.sum() > 0 and classifiers["subcategory"]["model"] is not None:
            subcat_metrics = evaluate_classifier(
                classifiers["subcategory"]["model"],
                classifiers["subcategory"]["encoder"],
                emb[subcat_mask],
                df.loc[subcat_mask, "subcategory"].values,
            )
            results[split_name]["subcategory"] = subcat_metrics

            cm_subcat = compute_confusion_matrix(
                classifiers["subcategory"]["model"],
                classifiers["subcategory"]["encoder"],
                emb[subcat_mask],
                df.loc[subcat_mask, "subcategory"].values,
            )
            if cm_subcat is not None:
                cm_subcat.to_csv(os.path.join(reports_dir, f"confusion_subcategory_{split_name}.csv"))
        else:
            results[split_name]["subcategory"] = {"accuracy": None, "macro_f1": None, "n_samples": 0}

        # Type (eligible rows only)
        type_mask = get_eligible_type_mask(df)
        if type_mask.sum() > 0 and classifiers["type"]["model"] is not None:
            type_metrics = evaluate_classifier(
                classifiers["type"]["model"],
                classifiers["type"]["encoder"],
                emb[type_mask],
                df.loc[type_mask, "type"].values,
            )
            results[split_name]["type"] = type_metrics

            cm_type = compute_confusion_matrix(
                classifiers["type"]["model"],
                classifiers["type"]["encoder"],
                emb[type_mask],
                df.loc[type_mask, "type"].values,
            )
            if cm_type is not None:
                cm_type.to_csv(os.path.join(reports_dir, f"confusion_type_{split_name}.csv"))
        else:
            results[split_name]["type"] = {"accuracy": None, "macro_f1": None, "n_samples": 0}

    return results


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty print evaluation metrics."""
    for split_name, split_metrics in metrics.items():
        print(f"\n=== {split_name.upper()} ===")
        for clf_name, clf_metrics in split_metrics.items():
            if clf_metrics["n_samples"] > 0:
                print(f"  {clf_name}:")
                print(f"    Accuracy: {clf_metrics['accuracy']}")
                print(f"    Macro-F1: {clf_metrics['macro_f1']}")
                print(f"    Samples:  {clf_metrics['n_samples']}")
            else:
                print(f"  {clf_name}: No eligible samples")
