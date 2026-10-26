"""
src/compare_results.py
======================
Compare a pipeline run with the published results of the paper.

Reads the result files of two run folders with the same layout — usually a
fresh run (--run) and the published results from the GitHub release
(--reference) — and compares every value behind the paper's tables, the
per-track test predictions of every model, and optionally the feature table
and the fold assignment. Nothing is written unless --details-csv is given.

Each value is reported as identical, within --tol, or different. Latency is
listed for information only, because it depends on the hardware. The exit
status is 0 when every compared value is within the tolerance and 1
otherwise.

The published results were produced on Linux x86-64 (CPU only) in --test-run
mode. A --test-run on the same platform and package versions is expected to
match exactly; on other platforms some values can differ in the third or
fourth decimal.

Compared (when present in both folders)
---------------------------------------
  Table 3          phase3_model_comparison.csv, per-model results files
  Table 4          cross_farm/cross_farm_results.csv
  Table 5          ablation/domain_ablation_results.csv
  Table 6          ablation/domain_only_results.csv
  Supp. Table 8    selected hyperparameters in the per-model results files
  Supp. Table 9    cross_farm_threshold_study/cross_farm_threshold_summary.csv
  Supp. Table 10   operating points (from the XGBoost test predictions)
  Supp. Table 11   convtran_batch_size_study/ (summary or bs_<N>/ folders)
  Supp. Table 12   ablation/group_loo_ablation.csv
  Supp. Table 13   ablation/feature_importance.csv

Usage
-----
  python src/compare_results.py \\
      --run       data/results/paper \\
      --reference TRAIT_paper_results \\
      [--features] [--splits-run data/splits_folds.csv \\
                    --splits-reference TRAIT_paper_results/splits_folds.csv] \\
      [--tol 0.0005] [--details-csv comparison.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.operating_points import operating_points, DEFAULT_THRESHOLDS

# (paper item, file, key column, value columns, required)
TABLES = [
    ("Table 3", "phase3_model_comparison.csv", "Model",
     ["pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi", "f1", "recall", "precision", "n_features"], True),
    ("Table 3 / Supp. Table 8", "prauc_protocol/prauc_protocol_comparison.csv", "model",
     ["threshold", "best_val_pr_auc", "best_C", "selected_n_estimators"], True),
    ("Table 3 / Supp. Table 8", "fcn_prauc/fcn_results.csv", "model",
     ["threshold", "selected_epochs", "validation_selection_score", "pr_auc", "f1"], True),
    ("Table 3 / Supp. Table 8", "rocket_prauc/rocket_results.csv", "model",
     ["threshold", "selected_alpha", "selected_alpha_val_pr_auc", "pr_auc", "f1"], True),
    ("Table 3 / Supp. Table 8", "convtran_prauc/convtran_results.csv", "model",
     ["threshold", "selected_epochs", "n_params", "validation_selection_score", "pr_auc", "f1"], True),
    ("Table 4", "cross_farm/cross_farm_results.csv", "evaluation",
     ["n_train", "n_test", "threshold", "pr_auc", "f1", "recall", "precision"], True),
    ("Table 5", "ablation/domain_ablation_results.csv", "label",
     ["n_features", "pr_auc", "f1", "recall", "precision"], True),
    ("Table 6", "ablation/domain_only_results.csv", "domain",
     ["n_features", "pr_auc", "f1", "recall", "precision", "threshold"], True),
    ("Supp. Table 9", "cross_farm_threshold_study/cross_farm_threshold_summary.csv", "evaluation",
     ["deployed_f1", "local_optimal_threshold", "local_optimal_f1", "f1_gap_deployed_vs_local_optimal"], False),
    ("Supp. Table 12", "ablation/group_loo_ablation.csv", "group", ["pr_auc", "f1"], True),
    ("Supp. Table 13", "ablation/feature_importance.csv", "feature",
     ["xgb_gain", "perm_pr_auc_drop", "perm_f1_drop"], True),
]

PREDICTIONS = [
    ("Majority baseline", "trivial/majority_predictions.csv"),
    ("Logistic Regression", "prauc_protocol/logistic_regression_prauc/predictions.csv"),
    ("Random Forest", "prauc_protocol/random_forest_prauc/predictions.csv"),
    ("XGBoost", "prauc_protocol/xgboost_prauc/predictions.csv"),
    ("1D-CNN", "fcn_prauc/fcn_predictions.csv"),
    ("miniROCKET", "rocket_prauc/rocket_predictions.csv"),
    ("ConvTran", "convtran_prauc/convtran_predictions.csv"),
]


def _read(folder: Path, rel: str) -> pd.DataFrame | None:
    """Read a CSV from a run folder, or return None if it does not exist."""
    path = folder / rel
    return pd.read_csv(path) if path.exists() else None


def _batch_size_table(folder: Path) -> pd.DataFrame | None:
    """Supp. Table 11 rows: the sweep summary, or one row per bs_<N>/ results file."""
    study = folder / "convtran_batch_size_study"
    summary = study / "convtran_batch_size_summary.csv"
    if summary.exists():
        return pd.read_csv(summary)
    rows = []
    for path in sorted(study.glob("bs_*/convtran_results.csv")):
        r = pd.read_csv(path).iloc[0].to_dict()
        r["batch_size"] = int(path.parent.name.split("_")[1])
        rows.append(r)
    return pd.DataFrame(rows) if rows else None


def _operating_point_table(folder: Path) -> pd.DataFrame | None:
    """Supp. Table 10 rows, recomputed from the saved XGBoost test predictions."""
    model_dir = folder / "prauc_protocol" / "xgboost_prauc"
    if not (model_dir / "predictions.csv").exists():
        return None
    preds = pd.read_csv(model_dir / "predictions.csv")
    deployed = float(json.loads((model_dir / "results.json").read_text())["threshold"])
    return operating_points(preds["y_true"].to_numpy(dtype=int), preds["score"].to_numpy(dtype=float),
                            DEFAULT_THRESHOLDS, deployed)


def compare_frames(item: str, name: str, ref: pd.DataFrame, run: pd.DataFrame,
                   key: str, columns: list[str], tol: float) -> list[dict]:
    """Compare the rows with the same key column by column."""
    out = []
    ref, run = ref.set_index(key), run.set_index(key)
    for row in ref.index:
        if row not in run.index:
            out.append({"item": item, "file": name, "row": row, "column": "", "reference": "",
                        "run": "", "difference": "", "status": "row only in reference"})
            continue
        for col in columns:
            if col not in ref.columns or col not in run.columns:
                continue
            a, b = ref.at[row, col], run.at[row, col]
            if pd.isna(a) and pd.isna(b):
                status, diff = "identical", 0.0
            else:
                try:
                    diff = float(b) - float(a)
                except (TypeError, ValueError):
                    status, diff = ("identical" if str(a) == str(b) else "different"), ""
                else:
                    status = ("identical" if diff == 0 else
                              "within tolerance" if abs(diff) <= tol else "different")
            out.append({"item": item, "file": name, "row": row, "column": col, "reference": a,
                        "run": b, "difference": diff, "status": status})
    return out


def compare_predictions(ref_dir: Path, run_dir: Path, tol: float) -> list[dict]:
    """Per-track test scores and predicted labels of each model."""
    out = []
    for model, rel in PREDICTIONS:
        ref, run = _read(ref_dir, rel), _read(run_dir, rel)
        if ref is None or run is None:
            continue
        m = ref.merge(run, on="filename", suffixes=("_ref", "_run"))
        dscore = (m["score_run"] - m["score_ref"]).abs()
        n_label = int((m["y_pred_ref"] != m["y_pred_run"]).sum())
        same_tracks = len(m) == len(ref) == len(run)
        status = ("different" if not same_tracks or n_label else
                  "identical" if dscore.max() == 0 else
                  "within tolerance" if dscore.max() <= tol else "different")
        out.append({"item": "Test predictions", "file": rel, "row": model,
                    "column": f"{len(m)} of {len(ref)} tracks; labels differing: {n_label}",
                    "reference": "", "run": "", "difference": float(dscore.max()), "status": status})
    return out


def compare_features(ref_dir: Path, run_dir: Path) -> list[dict]:
    """Feature table: number of values that are not equal (to 1e-9 relative)."""
    ref, run = _read(ref_dir, "track_features.csv"), _read(run_dir, "track_features.csv")
    if ref is None or run is None:
        return []
    m = ref.merge(run, on="filename", suffixes=("_ref", "_run"))
    cols = [c for c in ref.columns if c not in ("filename", "label") and f"{c}_run" in m.columns]
    a = m[[f"{c}_ref" for c in cols]].to_numpy(dtype=float)
    b = m[[f"{c}_run" for c in cols]].to_numpy(dtype=float)
    differ = ~np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=True)
    status = "identical" if not differ.any() and len(m) == len(ref) == len(run) else "different"
    return [{"item": "Feature table", "file": "track_features.csv", "row": f"{len(m)} tracks × {len(cols)} features",
             "column": f"values differing: {int(differ.sum())}", "reference": "", "run": "",
             "difference": "", "status": status}]


def compare_splits(ref_csv: Path, run_csv: Path) -> list[dict]:
    """Fold assignment of every track (Table 1 and every result depend on it)."""
    ref, run = pd.read_csv(ref_csv), pd.read_csv(run_csv)
    m = ref.merge(run, on="filename", suffixes=("_ref", "_run"))
    n_diff = int((m["fold_ref"] != m["fold_run"]).sum())
    status = "identical" if n_diff == 0 and len(m) == len(ref) == len(run) else "different"
    return [{"item": "Table 1 / folds", "file": run_csv.name, "row": f"{len(m)} tracks",
             "column": f"tracks in a different fold: {n_diff}", "reference": "", "run": "",
             "difference": "", "status": status}]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    p = argparse.ArgumentParser(description="Compare a pipeline run with the published results.")
    p.add_argument("--run", required=True, help="Run folder to check, e.g. data/results/paper")
    p.add_argument("--reference", required=True, help="Published results folder (unzipped release asset)")
    p.add_argument("--tol", type=float, default=0.0005,
                   help="Largest absolute difference counted as a match (default: 0.0005)")
    p.add_argument("--features", action="store_true", help="Also compare the 55 MB feature table")
    p.add_argument("--splits-run", default=None, help="Fold file of the run, e.g. data/splits_folds.csv")
    p.add_argument("--splits-reference", default=None, help="Published fold file")
    p.add_argument("--details-csv", default=None, help="Write every comparison to this CSV")
    return p.parse_args()


def main() -> int:
    """Run all comparisons, print a summary and return the exit status."""
    args = parse_args()
    ref_dir, run_dir = Path(args.reference).expanduser(), Path(args.run).expanduser()
    for d in (ref_dir, run_dir):
        if not d.is_dir():
            raise SystemExit(f"Folder not found: {d}")

    rows, skipped = [], []
    for item, rel, key, cols, required in TABLES:
        ref, run = _read(ref_dir, rel), _read(run_dir, rel)
        if ref is None or run is None:
            if ref is not None and required:
                rows.append({"item": item, "file": rel, "row": "", "column": "", "reference": "",
                             "run": "", "difference": "", "status": "missing in run"})
            else:
                skipped.append(f"{item} ({rel})")
            continue
        rows += compare_frames(item, rel, ref, run, key, cols, args.tol)

    for item, loader, key, cols in [
        ("Supp. Table 10", _operating_point_table, "threshold",
         ["precision", "recall", "f1", "missed_insect_pct", "false_positives"]),
        ("Supp. Table 11", _batch_size_table, "batch_size", ["pr_auc", "pr_auc_lo", "pr_auc_hi", "f1", "selected_epochs"]),
    ]:
        ref, run = loader(ref_dir), loader(run_dir)
        if ref is None or run is None:
            skipped.append(item)
            continue
        rows += compare_frames(item, item, ref, run, key, cols, args.tol)

    rows += compare_predictions(ref_dir, run_dir, args.tol)
    if args.features:
        rows += compare_features(ref_dir, run_dir)
    if args.splits_run and args.splits_reference:
        rows += compare_splits(Path(args.splits_reference), Path(args.splits_run))

    df = pd.DataFrame(rows)
    print(f"\nRun       : {run_dir}\nReference : {ref_dir}\nTolerance : {args.tol}\n")
    summary = (df.groupby(["item", "status"], sort=False).size().unstack(fill_value=0)
               .reindex(columns=["identical", "within tolerance", "different",
                                 "row only in reference", "missing in run"], fill_value=0))
    extra = [c for c in ("row only in reference", "missing in run") if not summary[c].any()]
    print(summary.drop(columns=extra).to_string())
    preds = df[df["item"] == "Test predictions"]
    if not preds.empty:
        print("\nPer-track test predictions (largest score difference):")
        for r in preds.itertuples():
            print(f"  {r.row:<20} {r.column:<38} max |Δscore| = {r.difference:.2e}  {r.status}")
    other = df[df["item"].isin(["Feature table", "Table 1 / folds"])]
    for r in other.itertuples():
        print(f"\n{r.item}: {r.row}, {r.column} — {r.status}")

    ref_lat, run_lat = _read(ref_dir, "phase3_model_comparison.csv"), _read(run_dir, "phase3_model_comparison.csv")
    if ref_lat is not None and run_lat is not None:
        lat = ref_lat[["Model", "latency_mean_ms"]].merge(run_lat[["Model", "latency_mean_ms"]],
                                                          on="Model", suffixes=(" reference", " run"))
        print("\nLatency in ms (depends on hardware, not compared):")
        print(lat.to_string(index=False))
    if skipped:
        print("\nNot compared (missing in one or both folders): " + "; ".join(skipped))

    bad = df[~df["status"].isin(["identical", "within tolerance"])]
    if not bad.empty:
        print(f"\nValues outside the tolerance ({len(bad)}):")
        for r in bad.head(40).itertuples():
            print(f"  {r.item:<24} {str(r.row):<34} {r.column:<26} reference={r.reference}  run={r.run}  ({r.status})")
        if len(bad) > 40:
            print(f"  … {len(bad) - 40} more (use --details-csv)")
    if args.details_csv:
        df.to_csv(args.details_csv, index=False)
        print(f"\nAll comparisons → {args.details_csv}")

    n_ok = int(df["status"].isin(["identical", "within tolerance"]).sum())
    print(f"\nResult: {'MATCH' if bad.empty else 'DIFFERENT'} — {n_ok} of {len(df)} comparisons within tolerance")
    return 0 if bad.empty else 1


if __name__ == "__main__":
    sys.exit(main())
