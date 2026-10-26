"""
src/cross_farm_threshold_study.py
=================================
Supplementary study — robustness of the decision threshold across farms
(Supp. Table 9).

Phase 4 (src/cross_farm.py) scores every within- and cross-farm evaluation
at the single threshold τ = 0.53 chosen on pooled training data. PR-AUC does
not depend on the threshold, but precision, recall and F1 do, and a fixed
threshold may land at a poor operating point when the score distribution or
the insect prevalence differs between farms.

For each of the four evaluations (A→A, B→B, A→B, B→A) this script retrains
XGBoost exactly as Phase 4 does (same hyperparameters, seed and imputation)
and reports precision, recall and F1 at the fixed threshold and at that
evaluation's own F1-optimal threshold (a sweep from 0.05 to 0.95). The
difference shows how much would be gained by re-tuning the threshold per
farm. It writes to its own folder and does not change the Phase 4 results.

Outputs (under <output-dir>/cross_farm_threshold_study/)
--------------------------------------------------------
  cross_farm_threshold_summary.csv      — one row per evaluation
  cross_farm_threshold_predictions.csv  — per-track scores

Usage (after the pipeline has run)
----------------------------------
  python src/cross_farm_threshold_study.py \\
      --features-csv data/results/trait_results/track_features.csv \\
      --splits-csv   data/splits_folds.csv \\
      --output-dir   data/results/trait_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.data_loader import load_data, impute_train_eval
# Reused from the Phase 4 study so this analysis trains the identical XGBoost
# model (including its seed).
from src.cross_farm import (
    _params_from_xgb_result,
    _threshold_from_xgb_result,
    _train_xgb,
)

SWEEP_THRESHOLDS = np.round(np.linspace(0.05, 0.95, 181), 4)


# ---------------------------------------------------------------------------
# Split helpers. Same partitions as src/cross_farm.py, but each also returns
# the test filenames so per-track predictions can be saved.
# ---------------------------------------------------------------------------

def _within_farm_split_with_files(df, farm, feature_cols):
    """Train on one farm's folds 0–3, test on its fold 4; also return test filenames."""
    farm_df    = df[df["cross_farm_role"] == farm]
    train_mask = farm_df["split_role"] == "trainval"
    test_mask  = farm_df["split_role"] == "test"

    X_tr = farm_df.loc[train_mask, feature_cols].values.astype("float32")
    y_tr = farm_df.loc[train_mask, "y"].values
    X_te = farm_df.loc[test_mask,  feature_cols].values.astype("float32")
    y_te = farm_df.loc[test_mask,  "y"].values
    filenames_te = farm_df.loc[test_mask, "filename"].to_numpy()
    X_tr, X_te = impute_train_eval(X_tr, X_te)
    return X_tr, y_tr, X_te, y_te, filenames_te


def _cross_farm_split_with_files(df, train_farm, feature_cols):
    """Train on all tracks of train_farm, test on all tracks of the other farm; also return test filenames."""
    train_mask = df["cross_farm_role"] == train_farm
    test_mask  = ~train_mask

    X_tr = df.loc[train_mask, feature_cols].values.astype("float32")
    y_tr = df.loc[train_mask, "y"].values.astype(int)
    X_te = df.loc[test_mask,  feature_cols].values.astype("float32")
    y_te = df.loc[test_mask,  "y"].values.astype(int)
    filenames_te = df.loc[test_mask, "filename"].to_numpy()
    X_tr, X_te = impute_train_eval(X_tr, X_te)
    return X_tr, y_tr, X_te, y_te, filenames_te


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def _best_local_threshold(y_true, y_prob):
    """Return (threshold, precision, recall, f1) maximising F1 on this evaluation's
    own predictions — i.e. the threshold that *would* have been chosen if this
    farm/evaluation's test scores were used for threshold tuning instead of the
    pooled validation set."""
    best_thr, best_f1, best_p, best_r = 0.5, -1.0, 0.0, 0.0
    for t in SWEEP_THRESHOLDS:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
            best_p = precision_score(y_true, pred, zero_division=0)
            best_r = recall_score(y_true, pred, zero_division=0)
    return best_thr, best_p, best_r, best_f1


def _metrics_at(y_true, y_prob, threshold):
    """Precision, recall and F1 at a given threshold."""
    pred = (y_prob >= threshold).astype(int)
    return (
        precision_score(y_true, pred, zero_division=0),
        recall_score(y_true, pred, zero_division=0),
        f1_score(y_true, pred, zero_division=0),
    )


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_threshold_study(df, feature_cols, xgb_params, deployed_threshold, out_dir: Path):
    """Run the four evaluations at the fixed and locally optimal thresholds and save the results."""
    print("\n── Cross-farm threshold sensitivity (Supp. Table 9) ──────────")
    print(f"  Deployed (fixed, pooled-training-tuned) threshold: {deployed_threshold:.4f}")

    pred_rows, summary_rows = [], []

    def _evaluate(evaluation, train_farm, test_farm, X_tr, y_tr, X_te, y_te, filenames_te):
        """Train, score and record one evaluation."""
        model  = _train_xgb(X_tr, y_tr, xgb_params)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred_deployed = (y_prob >= deployed_threshold).astype(int)

        for fn, yt, sc, yp in zip(filenames_te, y_te, y_prob, y_pred_deployed):
            pred_rows.append({
                "evaluation": evaluation, "train": train_farm, "test": test_farm,
                "filename": fn, "y_true": int(yt), "score": float(sc),
                "y_pred_deployed_threshold": int(yp),
            })

        dep_p, dep_r, dep_f1 = _metrics_at(y_te, y_prob, deployed_threshold)
        local_thr, local_p, local_r, local_f1 = _best_local_threshold(y_te, y_prob)

        print(f"  [{evaluation:<20}] n_test={len(y_te):>5,}  n_insect={int(y_te.sum()):>4,}  "
              f"deployed(tau={deployed_threshold:.3f}): P={dep_p:.4f} R={dep_r:.4f} F1={dep_f1:.4f}  |  "
              f"local-optimal(tau={local_thr:.3f}): P={local_p:.4f} R={local_r:.4f} F1={local_f1:.4f}")

        summary_rows.append({
            "evaluation": evaluation, "train": train_farm, "test": test_farm,
            "n_test": int(len(y_te)), "n_insect": int(y_te.sum()),
            "insect_prevalence": round(float(y_te.mean()), 4),
            "deployed_threshold": round(float(deployed_threshold), 4),
            "deployed_precision": round(float(dep_p), 4),
            "deployed_recall":    round(float(dep_r), 4),
            "deployed_f1":        round(float(dep_f1), 4),
            "local_optimal_threshold": round(float(local_thr), 4),
            "local_optimal_precision": round(float(local_p), 4),
            "local_optimal_recall":    round(float(local_r), 4),
            "local_optimal_f1":        round(float(local_f1), 4),
            "f1_gap_deployed_vs_local_optimal": round(float(local_f1 - dep_f1), 4),
            "recall_gap_deployed_vs_local_optimal": round(float(local_r - dep_r), 4),
        })

    # Within-farm (baseline: threshold tuned and applied on the same distribution)
    for farm in ("Farm_A", "Farm_B"):
        X_tr, y_tr, X_te, y_te, fn_te = _within_farm_split_with_files(df, farm, feature_cols)
        if len(X_te) < 5 or len(np.unique(y_te)) < 2:
            print(f"  [within_{farm}] insufficient test samples (n={len(X_te)}) — skipping")
            continue
        _evaluate(f"within_{farm}", farm, farm, X_tr, y_tr, X_te, y_te, fn_te)

    # Cross-farm transfer
    for train_farm, test_farm in [("Farm_A", "Farm_B"), ("Farm_B", "Farm_A")]:
        X_tr, y_tr, X_te, y_te, fn_te = _cross_farm_split_with_files(df, train_farm, feature_cols)
        _evaluate(f"{train_farm}_to_{test_farm}", train_farm, test_farm, X_tr, y_tr, X_te, y_te, fn_te)

    pred_df    = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(summary_rows)

    study_dir = out_dir / "cross_farm_threshold_study"
    study_dir.mkdir(parents=True, exist_ok=True)
    pred_path    = study_dir / "cross_farm_threshold_predictions.csv"
    summary_path = study_dir / "cross_farm_threshold_summary.csv"
    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  {'Evaluation':<20} {'n_test':>7} {'prev':>6}  "
          f"{'dep.P':>6} {'dep.R':>6} {'dep.F1':>6}   "
          f"{'loc.tau':>7} {'loc.P':>6} {'loc.R':>6} {'loc.F1':>6}   {'F1 gap':>7}")
    print("  " + "─" * 100)
    for _, r in summary_df.iterrows():
        print(f"  {r['evaluation']:<20} {r['n_test']:>7} {r['insect_prevalence']:>6.3f}  "
              f"{r['deployed_precision']:>6.3f} {r['deployed_recall']:>6.3f} {r['deployed_f1']:>6.3f}   "
              f"{r['local_optimal_threshold']:>7.3f} {r['local_optimal_precision']:>6.3f} "
              f"{r['local_optimal_recall']:>6.3f} {r['local_optimal_f1']:>6.3f}   "
              f"{r['f1_gap_deployed_vs_local_optimal']:>+7.3f}")

    print(f"\n  Predictions → {pred_path}")
    print(f"  Summary     → {summary_path}")
    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-csv", default="data/results/trait_results/track_features.csv")
    p.add_argument("--splits-csv",   default="data/splits_folds.csv")
    p.add_argument("--output-dir",   default="data/results/trait_results")
    p.add_argument("--xgb-params",   default=None)
    p.add_argument("--xgb-threshold", type=float, default=None,
                   help="Fixed decision threshold from Phase 3 XGBoost results.")
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("TRAIT_N_JOBS", "-1")))
    return p.parse_args()


def main():
    """Load the data and the Phase 3 XGBoost settings, then run the study."""
    args    = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df, feature_cols, _ = load_data(args.features_csv, args.splits_csv)
    print(f"  {len(df):,} tracks · {len(feature_cols)} features")

    xgb_threshold = args.xgb_threshold
    if args.xgb_params:
        xgb_params = json.loads(args.xgb_params)
        if xgb_threshold is None:
            xgb_threshold = 0.5
    else:
        xgb_res_path = out_dir / "prauc_protocol" / "xgboost_prauc" / "results.json"
        if not xgb_res_path.exists():
            raise SystemExit(
                f"Phase 3 XGBoost result not found: {xgb_res_path}. "
                "Run Phase 3 first or pass --xgb-params explicitly."
            )
        xgb_params = _params_from_xgb_result(xgb_res_path)
        if xgb_threshold is None:
            xgb_threshold = _threshold_from_xgb_result(xgb_res_path)
        print(f"  Loaded Phase 3 XGBoost params from {xgb_res_path}")
    xgb_params["n_jobs"] = int(args.n_jobs)

    run_threshold_study(df, feature_cols, xgb_params, float(xgb_threshold), out_dir)
    print("\n✓ Cross-farm threshold sensitivity study complete")


if __name__ == "__main__":
    main()
