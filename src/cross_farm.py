"""
src/cross_farm.py
=================
Phase 4 — Within-farm and cross-farm generalisation of XGBoost (Table 4).

Retrains XGBoost with the hyperparameters selected in Phase 3 and scores it
at the Phase 3 decision threshold (τ = 0.53), with no per-farm re-tuning, no
domain adaptation and no shared normalisation:

  A→A (within) : train on Farm A folds 0–3, test on Farm A fold 4
  B→B (within) : train on Farm B folds 0–3, test on Farm B fold 4
  A→B (cross)  : train on all Farm A tracks, test on all Farm B tracks
  B→A (cross)  : train on all Farm B tracks, test on all Farm A tracks

Farm identity defines the train/test partitions only; it is never a model
feature.

Outputs (under <output-dir>/cross_farm/)
----------------------------------------
  cross_farm_results.csv   — one row per evaluation: sizes, threshold,
                             precision, recall, F1 (with CI), PR-AUC, MCC
  cross_farm_summary.json  — the same rows as JSON

Usage
-----
  python src/cross_farm.py \\
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.data_loader import load_data, get_cross_farm_split, impute_train_eval
from src.models.evaluate import compute_metrics, bootstrap_ci, save_results

SEED   = 42
N_BOOT = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_from_xgb_result(path: Path) -> dict:
    """Load effective XGBoost params from a saved result/config JSON."""
    with open(path) as f:
        data = json.load(f)
    params = dict(data.get("best_params", data))
    selected_n = data.get("selected_n_estimators")
    if selected_n is not None:
        params["n_estimators"] = int(selected_n)
    return params


def _threshold_from_xgb_result(path: Path, default: float = 0.5) -> float:
    """Load the deployed XGBoost decision threshold from a saved result JSON."""
    with open(path) as f:
        data = json.load(f)
    try:
        return float(data.get("threshold", default))
    except (TypeError, ValueError):
        return float(default)


def _pos_weight(y):
    """Negative-to-positive class ratio, used as XGBoost's scale_pos_weight."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def _train_xgb(X_tr, y_tr, params):
    """Fit XGBoost with the given hyperparameters and a class-balancing scale_pos_weight."""
    import xgboost as xgb
    spw = _pos_weight(y_tr)
    params = dict(params)
    params.setdefault("n_jobs", int(os.environ.get("TRAIT_N_JOBS", "-1")))
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=spw,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=SEED, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def _within_farm_split(df, farm, feature_cols):
    """Train on trainval rows of a given farm, test on test rows of same farm."""
    farm_df    = df[df["cross_farm_role"] == farm]
    train_mask = farm_df["split_role"] == "trainval"
    test_mask  = farm_df["split_role"] == "test"

    X_tr = farm_df.loc[train_mask, feature_cols].values.astype("float32")
    y_tr = farm_df.loc[train_mask, "y"].values
    X_te = farm_df.loc[test_mask,  feature_cols].values.astype("float32")
    y_te = farm_df.loc[test_mask,  "y"].values
    # Impute NaNs using this farm's training-row medians only (no leakage).
    X_tr, X_te = impute_train_eval(X_tr, X_te)
    return X_tr, y_tr, X_te, y_te


def _run_evaluations(df, feature_cols, xgb_params, xgb_threshold):
    """Run the two within-farm and two cross-farm evaluations."""
    rows = []
    threshold = float(xgb_threshold)

    # ── Within-farm evaluations ──────────────────────────────────────────────
    for farm in ("Farm_A", "Farm_B"):
        X_tr, y_tr, X_te, y_te = _within_farm_split(df, farm, feature_cols)

        if len(X_te) < 5 or len(np.unique(y_te)) < 2:
            print(f"  {farm} within-farm: insufficient test samples "
                  f"(n={len(X_te)}) — skipping")
            rows.append({
                "evaluation": f"within_{farm}",
                "train": farm, "test": farm,
                "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                "threshold": round(threshold, 4),
                "threshold_mode": "phase3_xgb_fixed",
                "f1": None, "precision": None, "recall": None,
                "pr_auc": None, "mcc": None,
                "f1_lo": None, "f1_hi": None, "note": "too_few_test_samples",
            })
            continue

        model  = _train_xgb(X_tr, y_tr, xgb_params)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        m  = compute_metrics(y_te, y_pred, y_prob)
        ci = bootstrap_ci(y_te, y_pred, y_prob, n_boot=N_BOOT, seed=SEED)

        print(f"  Within {farm:<8}  n_train={len(X_tr):>5,}  "
              f"n_test={len(X_te):>4,}  F1={m['f1']:.4f} "
              f"[{ci['f1']['lo']:.4f}–{ci['f1']['hi']:.4f}]  "
              f"prec={m['precision']:.4f}  rec={m['recall']:.4f}")

        rows.append({
            "evaluation": f"within_{farm}",
            "train": farm, "test": farm,
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "threshold": round(threshold, 4),
            "threshold_mode": "phase3_xgb_fixed",
            **{k: m.get(k) for k in ("f1", "precision", "recall", "pr_auc", "mcc")},
            "f1_lo": ci["f1"]["lo"], "f1_hi": ci["f1"]["hi"],
            "note": "",
        })

    # ── Cross-farm: A→B and B→A ──────────────────────────────────────────────
    for train_farm, test_farm in [("Farm_A", "Farm_B"), ("Farm_B", "Farm_A")]:
        X_tr, y_tr, X_te, y_te = get_cross_farm_split(df, train_farm, feature_cols)

        model  = _train_xgb(X_tr, y_tr, xgb_params)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        m  = compute_metrics(y_te, y_pred, y_prob)
        ci = bootstrap_ci(y_te, y_pred, y_prob, n_boot=N_BOOT, seed=SEED)

        print(f"  {train_farm} → {test_farm}  "
              f"n_train={len(X_tr):>5,}  n_test={len(X_te):>4,}  "
              f"F1={m['f1']:.4f} [{ci['f1']['lo']:.4f}–{ci['f1']['hi']:.4f}]  "
              f"prec={m['precision']:.4f}  rec={m['recall']:.4f}")

        rows.append({
            "evaluation": f"{train_farm}_to_{test_farm}",
            "train": train_farm, "test": test_farm,
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "threshold": round(threshold, 4),
            "threshold_mode": "phase3_xgb_fixed",
            **{k: m.get(k) for k in ("f1", "precision", "recall", "pr_auc", "mcc")},
            "f1_lo": ci["f1"]["lo"], "f1_hi": ci["f1"]["hi"],
            "note": "",
        })

    return rows


# ---------------------------------------------------------------------------
# Main cross-farm routine
# ---------------------------------------------------------------------------

def run_cross_farm(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    xgb_threshold: float,
    out_dir: Path,
) -> pd.DataFrame:
    """Run the four within- and cross-farm evaluations and save the results.

    Farm identity is used only to define the train/test split, never as a
    feature.
    """
    print("\n── Phase 4  Cross-farm generalisation ─────────────────────────")
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed — skipping"); return pd.DataFrame()

    result_df = pd.DataFrame(
        _run_evaluations(df, feature_cols, xgb_params, xgb_threshold=xgb_threshold)
    )

    # ── Generalisation gap summary ────────────────────────────────────────────
    print("\n  Generalisation gap summary:")
    for train_farm, test_farm in [("Farm_A", "Farm_B"), ("Farm_B", "Farm_A")]:
        within_row = result_df[result_df["evaluation"] == f"within_{train_farm}"]
        cross_row  = result_df[result_df["evaluation"] == f"{train_farm}_to_{test_farm}"]
        if within_row.empty or cross_row.empty:
            continue
        w_f1 = within_row.iloc[0]["f1"]
        c_f1 = cross_row.iloc[0]["f1"]
        if w_f1 is None or c_f1 is None:
            continue
        gap     = round(float(w_f1) - float(c_f1), 4)
        pct_gap = round(100 * gap / float(w_f1), 2) if w_f1 else 0
        flag    = "✓ GOOD" if pct_gap < 6 else ("⚠ ACCEPTABLE" if pct_gap < 10 else "✗ LARGE")
        print(f"  {train_farm}→{test_farm}:  within={w_f1:.4f}  "
              f"cross={c_f1:.4f}  gap={gap:+.4f} ({pct_gap:.1f}%)  {flag}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_cf = out_dir / "cross_farm"
    out_cf.mkdir(parents=True, exist_ok=True)
    out_path = out_cf / "cross_farm_results.csv"
    result_df.to_csv(out_path, index=False)
    save_results(result_df.to_dict(orient="records"), out_cf, "cross_farm_summary")
    print(f"\n  Saved → {out_path.name}")
    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line options."""
    p = argparse.ArgumentParser(description="Phase 4 — Cross-farm generalisation")
    p.add_argument("--features-csv", default="data/results/trait_results/track_features.csv")
    p.add_argument("--splits-csv",   default="data/splits_folds.csv")
    p.add_argument("--output-dir",   default="data/results/trait_results")
    p.add_argument("--xgb-params",   default=None)
    p.add_argument("--xgb-threshold", type=float, default=None,
                   help="Fixed decision threshold from Phase 3 XGBoost results.")
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("TRAIT_N_JOBS", "-1")),
                   help="Parallel jobs for XGBoost. Use 1 for final reproducible runs.")
    return p.parse_args()


def main():
    """Load the data and the Phase 3 XGBoost settings, then run the evaluations."""
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

    print(f"  Using fixed XGBoost decision threshold = {float(xgb_threshold):.4f}")
    run_cross_farm(df, feature_cols, xgb_params, float(xgb_threshold), out_dir)
    print("\n✓ Phase 4 cross-farm complete")


if __name__ == "__main__":
    main()
