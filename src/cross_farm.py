"""
src/cross_farm.py
===========================
Phase 5 — Cross-farm generalisation experiments.

Evaluations (intrinsic-motion features only)
--------------------------------------------
  Within Farm A  : train folds 0-3 of Farm A, test fold 4 of Farm A
  Within Farm B  : train folds 0-3 of Farm B, test fold 4 of Farm B
  Farm A → Farm B: train all Farm A, test all Farm B
  Farm B → Farm A: train all Farm B, test all Farm A

No metadata-derived feature (time-of-day, station ID, season/farm) is used as
a model input. Farm identity defines the train/test split only — never a
feature — so cross-farm transfer measures generalisation of intrinsic motion.

Usage
-----
  python src/cross_farm.py \\
      --features-csv data/track_features.csv \\
      --splits-csv   data/splits_folds.csv \\
      --output-dir   data/results/six_domain_v1
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

# Metadata-derived column name kept only so it can be defensively dropped if a
# stale feature CSV still contains it. It is never used as a model feature.
TIME_OF_DAY_FEATURE = "time_of_day_bucket"


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
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def _train_xgb(X_tr, y_tr, params):
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


def _run_one_variant(df, feature_cols, xgb_params, xgb_threshold, variant_label):
    """Run all four cross-farm evaluations for a given feature set."""
    rows = []
    threshold = float(xgb_threshold)

    # ── Within-farm evaluations ──────────────────────────────────────────────
    for farm in ("Farm_A", "Farm_B"):
        X_tr, y_tr, X_te, y_te = _within_farm_split(df, farm, feature_cols)

        if len(X_te) < 5 or len(np.unique(y_te)) < 2:
            print(f"  [{variant_label}] {farm} within-farm: insufficient test samples "
                  f"(n={len(X_te)}) — skipping")
            rows.append({
                "variant": variant_label,
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

        print(f"  [{variant_label}] Within {farm:<8}  n_train={len(X_tr):>5,}  "
              f"n_test={len(X_te):>4,}  F1={m['f1']:.4f} "
              f"[{ci['f1']['lo']:.4f}–{ci['f1']['hi']:.4f}]  "
              f"prec={m['precision']:.4f}  rec={m['recall']:.4f}")

        rows.append({
            "variant": variant_label,
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

        print(f"  [{variant_label}] {train_farm} → {test_farm}  "
              f"n_train={len(X_tr):>5,}  n_test={len(X_te):>4,}  "
              f"F1={m['f1']:.4f} [{ci['f1']['lo']:.4f}–{ci['f1']['hi']:.4f}]  "
              f"prec={m['precision']:.4f}  rec={m['recall']:.4f}")

        rows.append({
            "variant": variant_label,
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
    """
    Run cross-farm generalisation experiments on the standard 125-feature set.

    No metadata-derived feature (time-of-day, station ID, season/farm) is used
    as a model input. Farm identity is used only to define the train/test
    split, never as a feature.
    """
    print("\n── Phase 5  Cross-farm generalisation ─────────────────────────")
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed — skipping"); return pd.DataFrame()

    # Defensively drop any metadata-derived column that may linger in a stale CSV.
    feature_cols = [f for f in feature_cols if f != TIME_OF_DAY_FEATURE]

    all_rows = []

    # ── Standard feature set (intrinsic motion only) ────────────────────────
    print("\n  Variant: intrinsic motion features only")
    all_rows.extend(
        _run_one_variant(
            df,
            feature_cols,
            xgb_params,
            xgb_threshold=xgb_threshold,
            variant_label="no_diel",
        )
    )

    result_df = pd.DataFrame(all_rows)

    # ── Generalisation gap summary ────────────────────────────────────────────
    print("\n  Generalisation gap summary (variant: no_diel):")
    no_diel = result_df[result_df["variant"] == "no_diel"]
    for train_farm, test_farm in [("Farm_A", "Farm_B"), ("Farm_B", "Farm_A")]:
        within_row = no_diel[no_diel["evaluation"] == f"within_{train_farm}"]
        cross_row  = no_diel[no_diel["evaluation"] == f"{train_farm}_to_{test_farm}"]
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

    # ── Diel feature contribution ─────────────────────────────────────────────
    if "with_diel" in result_df["variant"].values:
        print("\n  Diel feature contribution (with_diel - no_diel):")
        for eval_name in result_df["evaluation"].unique():
            nd = result_df[(result_df["variant"] == "no_diel") &
                           (result_df["evaluation"] == eval_name)]
            wd = result_df[(result_df["variant"] == "with_diel") &
                           (result_df["evaluation"] == eval_name)]
            if nd.empty or wd.empty:
                continue
            nd_f1 = nd.iloc[0]["f1"]
            wd_f1 = wd.iloc[0]["f1"]
            if nd_f1 is None or wd_f1 is None:
                continue
            delta = round(float(wd_f1) - float(nd_f1), 4)
            print(f"  {eval_name:<35}  Δ={delta:+.4f}")

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
    p = argparse.ArgumentParser(description="Phase 5 — Cross-farm generalisation")
    p.add_argument("--features-csv", default="data/track_features.csv")
    p.add_argument("--splits-csv",   default="data/splits_folds.csv")
    p.add_argument("--audit-csv",    default=None)
    p.add_argument("--output-dir",   default="data/results/six_domain_v1")
    p.add_argument("--xgb-params",   default=None)
    p.add_argument("--xgb-threshold", type=float, default=None,
                   help="Fixed decision threshold from Phase 3 XGBoost results.")
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("TRAIT_N_JOBS", "-1")),
                   help="Parallel jobs for XGBoost. Use 1 for final reproducible runs.")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df, feature_cols, _ = load_data(args.features_csv, args.splits_csv, args.audit_csv)
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
                f"Phase 2 XGBoost PR-AUC result not found: {xgb_res_path}. "
                "Run Phase 2 first or pass --xgb-params explicitly."
            )
        xgb_params = _params_from_xgb_result(xgb_res_path)
        if xgb_threshold is None:
            xgb_threshold = _threshold_from_xgb_result(xgb_res_path)
        print(f"  Loaded Phase 2 XGBoost PR-AUC params from {xgb_res_path}")
    xgb_params["n_jobs"] = int(args.n_jobs)

    print(f"  Using fixed XGBoost decision threshold = {float(xgb_threshold):.4f}")
    run_cross_farm(df, feature_cols, xgb_params, float(xgb_threshold), out_dir)
    print("\n✓ Phase 5 cross-farm complete")


if __name__ == "__main__":
    main()
