"""
Evaluate AI classifier comparison outputs and build/score regression sets.

Primary mode:
- Read comparison_ai.csv and print sector/tag/exact metrics.
- Print confusion matrix, top confusions, per-sector precision/recall/F1.
- Print false positives/false negatives by sector and tag.
- Export row-level errors with audit fields.

Optional modes:
- Build regression fixture from high-impact error buckets.
- Compare baseline vs candidate on a fixture and print fixed/broken deltas.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SECTOR_COL_TRUE = "sector"
SECTOR_COL_PRED = "pred_sector"
TAG_COL_TRUE = "tag"
TAG_COL_PRED = "pred_tag"


def _load_comparison(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "description", SECTOR_COL_TRUE, SECTOR_COL_PRED, TAG_COL_TRUE, TAG_COL_PRED}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Comparison file missing required columns: {sorted(missing)}")

    df[TAG_COL_TRUE] = df[TAG_COL_TRUE].fillna("none").astype(str)
    df[TAG_COL_PRED] = df[TAG_COL_PRED].fillna("none").astype(str)
    df[SECTOR_COL_TRUE] = df[SECTOR_COL_TRUE].fillna("__MISSING__").astype(str)
    df[SECTOR_COL_PRED] = df[SECTOR_COL_PRED].fillna("__MISSING__").astype(str)

    df["sector_match"] = df[SECTOR_COL_TRUE] == df[SECTOR_COL_PRED]
    df["tag_match"] = df[TAG_COL_TRUE] == df[TAG_COL_PRED]
    df["exact_match"] = df["sector_match"] & df["tag_match"]
    return df


def _sector_metrics(df: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(set(df[SECTOR_COL_TRUE].unique()) | set(df[SECTOR_COL_PRED].unique()))
    rows: List[Dict] = []
    for label in labels:
        tp = int(((df[SECTOR_COL_TRUE] == label) & (df[SECTOR_COL_PRED] == label)).sum())
        fp = int(((df[SECTOR_COL_TRUE] != label) & (df[SECTOR_COL_PRED] == label)).sum())
        fn = int(((df[SECTOR_COL_TRUE] == label) & (df[SECTOR_COL_PRED] != label)).sum())
        support = int((df[SECTOR_COL_TRUE] == label).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "sector": label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def _confusion(df: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(set(df[SECTOR_COL_TRUE].unique()) | set(df[SECTOR_COL_PRED].unique()))
    cm = pd.crosstab(df[SECTOR_COL_TRUE], df[SECTOR_COL_PRED], rownames=["True"], colnames=["Pred"])
    return cm.reindex(index=labels, columns=labels, fill_value=0)


def _top_confusions(cm: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for true_sector in cm.index:
        for pred_sector in cm.columns:
            if true_sector == pred_sector:
                continue
            count = int(cm.loc[true_sector, pred_sector])
            if count > 0:
                rows.append(
                    {"true_sector": true_sector, "pred_sector": pred_sector, "count": count}
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("count", ascending=False).head(top_n)


def _tag_fp_fn(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fp = df[(df[TAG_COL_TRUE] != "garage") & (df[TAG_COL_PRED] == "garage")]
    fn = df[(df[TAG_COL_TRUE] == "garage") & (df[TAG_COL_PRED] != "garage")]
    return fp, fn


def _error_bucket(row: pd.Series) -> str:
    if not row["sector_match"] and not row["tag_match"]:
        return f"sector+tag:{row[SECTOR_COL_TRUE]}->{row[SECTOR_COL_PRED]}|{row[TAG_COL_TRUE]}->{row[TAG_COL_PRED]}"
    if not row["sector_match"]:
        return f"sector:{row[SECTOR_COL_TRUE]}->{row[SECTOR_COL_PRED]}"
    return f"tag:{row[TAG_COL_TRUE]}->{row[TAG_COL_PRED]}"


def _export_error_rows(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    errors = df[~df["exact_match"]].copy()
    errors["error_bucket"] = errors.apply(_error_bucket, axis=1)
    if "matched_rule_signals" not in errors.columns:
        errors["matched_rule_signals"] = ""

    cols = [
        "id",
        "description",
        SECTOR_COL_TRUE,
        TAG_COL_TRUE,
        SECTOR_COL_PRED,
        TAG_COL_PRED,
        "pred_sector_conf",
        "rationale",
        "error_bucket",
        "matched_rule_signals",
    ]
    out = errors[[c for c in cols if c in errors.columns]].copy()
    out = out.rename(
        columns={
            SECTOR_COL_TRUE: "true_sector",
            TAG_COL_TRUE: "true_tag",
            SECTOR_COL_PRED: "pred_sector",
            TAG_COL_PRED: "pred_tag",
            "pred_sector_conf": "confidence",
        }
    )
    out.to_csv(output_path, index=False)
    return out


def _build_regression_fixture(df: pd.DataFrame, fixture_path: Path) -> pd.DataFrame:
    def pick(true_sector: str, pred_sector: str, n: int = None) -> pd.DataFrame:
        x = df[(df[SECTOR_COL_TRUE] == true_sector) & (df[SECTOR_COL_PRED] == pred_sector)].copy()
        if n is not None:
            return x.head(n)
        return x

    buckets = [
        pick("Miscellaneous", "Self Build", 51),
        pick("Miscellaneous", "Residential", 17),
        pick("Miscellaneous", "Commercial & Retail", 16),
        pick("Miscellaneous", "Agriculture", 13),
        pick("Self Build", "Residential", 12),
        pick("Civil", "Miscellaneous", None),
        pick("Miscellaneous", "Civil", None),
        df[df["tag_match"] == False].copy(),
    ]

    fixture = pd.concat(buckets, ignore_index=True)
    fixture = fixture.drop_duplicates(subset=["id", "description"], keep="first")
    fixture["fixture_bucket"] = fixture.apply(_error_bucket, axis=1)

    keep_cols = [
        "id",
        "description",
        SECTOR_COL_TRUE,
        TAG_COL_TRUE,
        SECTOR_COL_PRED,
        TAG_COL_PRED,
        "pred_sector_conf",
        "rationale",
        "matched_rule_signals",
        "fixture_bucket",
    ]
    fixture = fixture[[c for c in keep_cols if c in fixture.columns]].copy()
    fixture = fixture.rename(
        columns={
            SECTOR_COL_TRUE: "true_sector",
            TAG_COL_TRUE: "true_tag",
            SECTOR_COL_PRED: "baseline_pred_sector",
            TAG_COL_PRED: "baseline_pred_tag",
        }
    )

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(fixture_path, index=False)
    return fixture


def _compare_regression(baseline_path: str, candidate_path: str, fixture_path: str) -> Dict:
    baseline = _load_comparison(baseline_path)
    candidate = _load_comparison(candidate_path)
    fixture = pd.read_csv(fixture_path)

    keys = ["id", "description"]
    fixture_keys = fixture[keys].drop_duplicates()

    b = baseline.merge(fixture_keys, on=keys, how="inner").copy()
    c = candidate.merge(fixture_keys, on=keys, how="inner").copy()

    merged = b[keys + [SECTOR_COL_TRUE, TAG_COL_TRUE, SECTOR_COL_PRED, TAG_COL_PRED]].merge(
        c[keys + [SECTOR_COL_PRED, TAG_COL_PRED]],
        on=keys,
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )

    merged["baseline_sector_ok"] = merged[SECTOR_COL_PRED + "_baseline"] == merged[SECTOR_COL_TRUE]
    merged["candidate_sector_ok"] = merged[SECTOR_COL_PRED + "_candidate"] == merged[SECTOR_COL_TRUE]

    fixed = merged[(~merged["baseline_sector_ok"]) & (merged["candidate_sector_ok"])].copy()
    broken = merged[(merged["baseline_sector_ok"]) & (~merged["candidate_sector_ok"])].copy()

    baseline_acc = float(merged["baseline_sector_ok"].mean()) if len(merged) else 0.0
    candidate_acc = float(merged["candidate_sector_ok"].mean()) if len(merged) else 0.0

    return {
        "fixture_rows": int(len(merged)),
        "fixed_count": int(len(fixed)),
        "broken_count": int(len(broken)),
        "net_accuracy_delta": round(candidate_acc - baseline_acc, 4),
        "newly_broken_rows": broken[
            [
                "id",
                "description",
                SECTOR_COL_TRUE,
                SECTOR_COL_PRED + "_baseline",
                SECTOR_COL_PRED + "_candidate",
            ]
        ].to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI comparison outputs and regression deltas")
    parser.add_argument("--comparison", help="Path to comparison_ai.csv for evaluation")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N sector confusions to print")
    parser.add_argument(
        "--error-export",
        default=None,
        help="Path to export row-level errors CSV (default: alongside comparison as error_rows_ai.csv)",
    )
    parser.add_argument(
        "--build-fixture",
        action="store_true",
        help="Build tests/fixtures/classifier_regression_cases.csv from --comparison",
    )
    parser.add_argument(
        "--fixture-path",
        default="tests/fixtures/classifier_regression_cases.csv",
        help="Fixture path for --build-fixture or --compare",
    )
    parser.add_argument("--baseline-comparison", help="Baseline comparison_ai.csv for regression compare")
    parser.add_argument("--candidate-comparison", help="Candidate comparison_ai.csv for regression compare")

    args = parser.parse_args()

    if args.baseline_comparison and args.candidate_comparison:
        result = _compare_regression(args.baseline_comparison, args.candidate_comparison, args.fixture_path)
        print("Regression comparison")
        print(f"- fixture_rows: {result['fixture_rows']}")
        print(f"- fixed_count: {result['fixed_count']}")
        print(f"- broken_count: {result['broken_count']}")
        print(f"- net_accuracy_delta: {result['net_accuracy_delta']}")
        print("- newly_broken_rows:")
        if not result["newly_broken_rows"]:
            print("  none")
        else:
            for row in result["newly_broken_rows"]:
                print(
                    f"  id={row['id']} true={row[SECTOR_COL_TRUE]} baseline={row[SECTOR_COL_PRED + '_baseline']} candidate={row[SECTOR_COL_PRED + '_candidate']}"
                )
        return

    if not args.comparison:
        raise ValueError("--comparison is required unless using --baseline-comparison and --candidate-comparison")

    df = _load_comparison(args.comparison)

    sector_acc = float(df["sector_match"].mean())
    tag_acc = float(df["tag_match"].mean())
    exact_acc = float(df["exact_match"].mean())

    cm = _confusion(df)
    top_conf = _top_confusions(cm, args.top_n)
    sector_metrics = _sector_metrics(df)

    misc_row = sector_metrics[sector_metrics["sector"] == "Miscellaneous"]
    self_build_row = sector_metrics[sector_metrics["sector"] == "Self Build"]
    misc_recall = float(misc_row.iloc[0]["recall"]) if len(misc_row) else np.nan
    self_build_recall = float(self_build_row.iloc[0]["recall"]) if len(self_build_row) else np.nan

    print("Overall metrics")
    print(f"- rows: {len(df)}")
    print(f"- sector_accuracy: {sector_acc:.4f}")
    print(f"- tag_accuracy: {tag_acc:.4f}")
    print(f"- exact_sector_tag_accuracy: {exact_acc:.4f}")
    print(f"- miscellaneous_recall: {misc_recall:.4f}")
    print(f"- self_build_recall: {self_build_recall:.4f}")

    print("\nSector confusion matrix")
    print(cm.to_string())

    print("\nTop sector confusions")
    if top_conf.empty:
        print("- none")
    else:
        for _, row in top_conf.iterrows():
            print(f"- {row['true_sector']} -> {row['pred_sector']}: {int(row['count'])}")

    print("\nPer-sector precision/recall/F1")
    print(sector_metrics.to_string(index=False))

    print("\nFalse positives / false negatives by sector")
    for _, row in sector_metrics.iterrows():
        print(f"- {row['sector']}: FP={int(row['fp'])}, FN={int(row['fn'])}")

    tag_fp, tag_fn = _tag_fp_fn(df)
    print("\nTag false positives / false negatives")
    print(f"- tag_false_positive_count: {len(tag_fp)}")
    print(f"- tag_false_negative_count: {len(tag_fn)}")

    error_export = Path(args.error_export) if args.error_export else (Path(args.comparison).parent / "error_rows_ai.csv")
    error_rows = _export_error_rows(df, error_export)
    print(f"\nExported row-level errors: {error_export}")
    print(f"- error_rows: {len(error_rows)}")

    if args.build_fixture:
        fixture_path = Path(args.fixture_path)
        fixture = _build_regression_fixture(df, fixture_path)
        print(f"Built regression fixture: {fixture_path}")
        print(f"- fixture_rows: {len(fixture)}")


if __name__ == "__main__":
    main()
