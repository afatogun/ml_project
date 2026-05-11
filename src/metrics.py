"""Evaluation metrics and reporting."""

import json
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .classifier_features import extract_classifier_features
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


def _predict_labels_and_probabilities(model, encoder, X: np.ndarray):
    y_pred_encoded = model.predict(X)
    y_pred = encoder.inverse_transform(y_pred_encoded)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

    return y_pred, probabilities


def _per_sector_metrics_df(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "sector": labels,
            "precision": np.round(precision, 4),
            "recall": np.round(recall, 4),
            "f1": np.round(f1, 4),
            "support": support.astype(int),
        }
    )


def _top_confusions_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    confusion_rows = pd.DataFrame({"true_sector": y_true, "pred_sector": y_pred})
    confusion_rows = confusion_rows[confusion_rows["true_sector"] != confusion_rows["pred_sector"]]
    if confusion_rows.empty:
        return pd.DataFrame(columns=["true_sector", "pred_sector", "count"])
    top_confusions = (
        confusion_rows.groupby(["true_sector", "pred_sector"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return top_confusions


def _confidence_buckets_df(confidence: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    bucket_edges = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    bucket_labels = ["0.00-0.49", "0.50-0.59", "0.60-0.69", "0.70-0.79", "0.80-0.89", "0.90-1.00"]
    bucket_series = pd.cut(confidence, bins=bucket_edges, labels=bucket_labels, right=False, include_lowest=True)
    bucket_df = pd.DataFrame({"bucket": bucket_series, "correct": correct.astype(int), "confidence": confidence})
    summary = (
        bucket_df.groupby("bucket", observed=False)
        .agg(n_rows=("correct", "size"), accuracy=("correct", "mean"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )
    summary["accuracy"] = summary["accuracy"].fillna(0).round(4)
    summary["avg_confidence"] = summary["avg_confidence"].fillna(0).round(4)
    return summary


def _wrong_rows_df(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray],
    labels: np.ndarray,
) -> pd.DataFrame:
    top_confidence = np.ones(len(y_pred), dtype=float)
    top_sector_json = [json.dumps([pred]) for pred in y_pred]
    top_prob_json = [json.dumps([1.0]) for _ in y_pred]

    if probabilities is not None:
        top_confidence = probabilities.max(axis=1)
        top_order = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
        top_sector_json = [
            json.dumps([str(labels[idx]) for idx in row])
            for row in top_order
        ]
        top_prob_json = [
            json.dumps([round(float(probabilities[row_index, idx]), 6) for idx in row])
            for row_index, row in enumerate(top_order)
        ]

    feature_json = [json.dumps(extract_classifier_features(desc)) for desc in df["description"].fillna("").astype(str)]

    audit_df = pd.DataFrame(
        {
            "id": df["id"].values,
            "description": df["description"].values,
            "true_sector": y_true,
            "pred_sector": y_pred,
            "model_confidence": np.round(top_confidence, 6),
            "top_3_predicted_sectors": top_sector_json,
            "top_3_predicted_probs": top_prob_json,
            "rule_features": feature_json,
            "nearest_labelled_examples": ["[]"] * len(df),
        }
    )
    return audit_df[audit_df["true_sector"] != audit_df["pred_sector"]].reset_index(drop=True)


def export_sector_diagnostics(
    model,
    encoder,
    X: np.ndarray,
    df: pd.DataFrame,
    reports_dir: str,
    split_name: str,
) -> Dict[str, Any]:
    if model is None:
        return {}

    if X is None:
        # Some wrappers (for example probability ensembles) already hold per-row
        # probabilities internally and ignore X. Provide a shape-safe placeholder.
        X = np.zeros((len(df), 1), dtype=np.float32)

    if len(X) == 0:
        return {}

    y_true = df["sector"].values
    labels = encoder.classes_
    y_pred, probabilities = _predict_labels_and_probabilities(model, encoder, X)
    top_confidence = probabilities.max(axis=1) if probabilities is not None else np.ones(len(y_pred), dtype=float)
    correct = (y_true == y_pred)

    per_sector_df = _per_sector_metrics_df(y_true, y_pred, labels)
    top_confusions_df = _top_confusions_df(y_true, y_pred)
    confidence_df = _confidence_buckets_df(top_confidence, correct)
    wrong_rows_df = _wrong_rows_df(df, y_true, y_pred, probabilities, labels)

    per_sector_path = os.path.join(reports_dir, f"sector_metrics_{split_name}.csv")
    top_confusions_path = os.path.join(reports_dir, f"sector_top_confusions_{split_name}.csv")
    confidence_path = os.path.join(reports_dir, f"sector_confidence_buckets_{split_name}.csv")
    wrong_rows_path = os.path.join(reports_dir, f"sector_wrong_rows_{split_name}.csv")

    per_sector_df.to_csv(per_sector_path, index=False)
    top_confusions_df.to_csv(top_confusions_path, index=False)
    confidence_df.to_csv(confidence_path, index=False)
    wrong_rows_df.to_csv(wrong_rows_path, index=False)

    return {
        "per_sector_metrics_csv": per_sector_path,
        "top_confusions_csv": top_confusions_path,
        "confidence_buckets_csv": confidence_path,
        "wrong_rows_csv": wrong_rows_path,
        "top_confusions": top_confusions_df.head(10).to_dict(orient="records"),
    }


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
        sector_metrics["diagnostics"] = export_sector_diagnostics(
            classifiers["sector"]["model"],
            classifiers["sector"]["encoder"],
            emb,
            df,
            reports_dir,
            split_name,
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

        # Tag (all rows, binary: "garage" vs "none")
        if classifiers.get("tag") and classifiers["tag"]["model"] is not None:
            tag_labels_all = df["tag"].fillna("none").values
            tag_metrics = evaluate_classifier(
                classifiers["tag"]["model"],
                classifiers["tag"]["encoder"],
                emb,
                tag_labels_all,
            )
            results[split_name]["tag"] = tag_metrics

            cm_tag = compute_confusion_matrix(
                classifiers["tag"]["model"],
                classifiers["tag"]["encoder"],
                emb,
                tag_labels_all,
            )
            if cm_tag is not None:
                cm_tag.to_csv(os.path.join(reports_dir, f"confusion_tag_{split_name}.csv"))
        else:
            results[split_name]["tag"] = {"accuracy": None, "macro_f1": None, "n_samples": 0}

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
