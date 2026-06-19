"""
src/models/majority_baseline.py
=============================
Step 2.1 — Train and evaluate baseline classifiers.

Models
------
  1. Trivial baselines (majority, random, single-feature threshold)
  2. Logistic Regression  — C grid search, 5-fold GroupKFold
  3. Random Forest        — grid search on n_estimators, max_depth
  4. XGBoost              — grid search with early stopping

Each model:
  • Is evaluated on the held-out test set (fold 4) with bootstrap CIs.
  • Reports inference latency (single-sample, 200 reps).
  • Saves a trained model artifact (.pkl or .ubj) and a results JSON.

Usage
-----
  python src/models/majority_baseline.py \\
      --features-csv data/track_features.csv \\
      --splits-csv   data/splits_folds.csv \\
      --audit-csv    data/data_audit_report.csv \\
      --output-dir   data/results/six_domain_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.data_loader import load_data, get_fold_split, get_test_split, FEATURE_GROUPS
from src.models.evaluate import (
    compute_metrics, bootstrap_ci, measure_latency, save_results, results_to_row
)

SEED = 42
N_BOOT = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(y):
    return np.where(y == 1, "insect", "not_insect")


def _scale(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test), sc


def _pos_weight(y):
    """XGBoost scale_pos_weight = count_neg / count_pos."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def save_pr_curve(y_true, y_score, model_name: str, out_dir: Path) -> Path:
    """Save precision-recall curve points for plotting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    threshold_col = np.r_[thresholds, np.nan]
    ap = average_precision_score(y_true, y_score)
    curve = pd.DataFrame({
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "threshold": threshold_col,
        "average_precision": ap,
        "n_total": len(y_true),
        "n_positive": int(np.sum(y_true)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_pr_curve.csv"
    curve.to_csv(path, index=False)
    return path


def save_predictions(df, y_true, y_pred, y_score, out_dir: Path, stem: str = "predictions") -> Path:
    """Save per-test-sample predictions (filename, y_true, y_pred, score).

    The row order of get_test_split's X_te matches df.loc[split_role == "test"],
    so test filenames are recovered in the same order.
    """
    test_fn = df.loc[df["split_role"] == "test", "filename"].tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    pd.DataFrame({
        "filename": test_fn,
        "y_true": np.asarray(y_true).astype(int),
        "y_pred": np.asarray(y_pred).astype(int),
        "score": np.asarray(y_score, dtype=float),
    }).to_csv(path, index=False)
    return path


def _cv_score(model, df, feature_cols, metric="f1"):
    """5-fold GroupKFold cross-validation F1 (on trainval only)."""
    scores = []
    for fold in range(4):  # folds 0-3 are trainval
        X_tr, y_tr, X_val, y_val = get_fold_split(df, fold, feature_cols)
        m = joblib.loads(joblib.dumps(model))  # clone
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)
        from sklearn.metrics import f1_score
        scores.append(f1_score(y_val, y_pred, zero_division=0))
    return float(np.mean(scores))


def run_majority_baseline(df, feature_cols, out_dir):
    """Run only the majority-class baseline."""
    print("\n── Majority baseline ─────────────────────────────────────────")
    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)

    maj = DummyClassifier(strategy="most_frequent", random_state=SEED)
    maj.fit(X_tr, y_tr)
    y_pred = maj.predict(X_te)
    y_score = maj.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, y_pred, y_score)
    ci = bootstrap_ci(y_te, y_pred, y_score, n_boot=N_BOOT, seed=SEED)
    lat = measure_latency(maj.predict, X_te[0])
    save_pr_curve(y_te, y_score, "majority", out_dir / "trivial")
    save_predictions(df, y_te, y_pred, y_score, out_dir / "trivial", stem="majority_predictions")
    results = {}
    results["majority"] = {"metrics": m, "ci": ci, "latency": lat}
    print(f"  majority   F1={m['f1']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}")
    save_results(results, out_dir / "trivial", "trivial_results")
    return results


# ---------------------------------------------------------------------------
# 1. Trivial baselines
# ---------------------------------------------------------------------------

def run_trivial(df, feature_cols, out_dir):
    print("\n── Trivial baselines ──────────────────────────────────────────")
    results = run_majority_baseline(df, feature_cols, out_dir)

    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)

    # Random (stratified)
    rand = DummyClassifier(strategy="stratified", random_state=SEED)
    rand.fit(X_tr, y_tr)
    y_pred = rand.predict(X_te)
    y_score = rand.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, y_pred, y_score)
    ci = bootstrap_ci(y_te, y_pred, y_score, n_boot=N_BOOT, seed=SEED)
    lat = measure_latency(rand.predict, X_te[0])
    save_pr_curve(y_te, y_score, "random_stratified", out_dir / "trivial")
    results["random_stratified"] = {"metrics": m, "ci": ci, "latency": lat}
    print(f"  random     F1={m['f1']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}")

    # Single-feature threshold (best feature by val-set F1)
    _single_feature_threshold(df, feature_cols, X_tr, y_tr, X_te, y_te, results, out_dir)

    save_results(results, out_dir / "trivial", "trivial_results")
    return results


def _single_feature_threshold(df, feature_cols, X_tr, y_tr, X_te, y_te, results, out_dir):
    """Find the best single-feature + threshold on trainval, evaluate on test."""
    from sklearn.metrics import f1_score

    # Candidate features
    candidates = [
        "convex_hull_area", "msd_alpha", "mean_reversion_strength",
        "oscillation_regularity_x", "track_confinement_ratio",
    ]
    best_name, best_f1, best_thresh, best_dir = None, -1, 0.5, "gt"

    for feat in candidates:
        if feat not in feature_cols:
            continue
        idx = feature_cols.index(feat)
        vals_tr = X_tr[:, idx]

        # Threshold search on trainval (using fold 3 as proxy val)
        X_tv, y_tv, X_val, y_val = get_fold_split(df, 3, feature_cols)
        v = X_val[:, idx]

        for direction in ("gt", "lt"):
            for pct in np.arange(5, 96, 5):
                thr = np.percentile(vals_tr, pct)
                pred = (v > thr).astype(int) if direction == "gt" else (v < thr).astype(int)
                f = f1_score(y_val, pred, zero_division=0)
                if f > best_f1:
                    best_f1, best_thresh, best_dir, best_name = f, thr, direction, feat

    if best_name is None:
        return

    idx = feature_cols.index(best_name)
    vals_te = X_te[:, idx]
    y_pred = (vals_te > best_thresh).astype(int) if best_dir == "gt" \
             else (vals_te < best_thresh).astype(int)
    y_score = vals_te if best_dir == "gt" else -vals_te
    m = compute_metrics(y_te, y_pred, y_score)
    ci = bootstrap_ci(y_te, y_pred, y_score, n_boot=N_BOOT, seed=SEED)
    save_pr_curve(y_te, y_score, "single_feature", out_dir / "trivial")
    results["single_feature"] = {
        "metrics": m, "ci": ci,
        "feature": best_name, "threshold": best_thresh, "direction": best_dir,
        "latency": {"latency_mean_ms": 0.01},
    }
    print(f"  single-feat({best_name})  F1={m['f1']:.3f}  "
          f"thr={best_thresh:.1f} ({best_dir})")


# ---------------------------------------------------------------------------
# 2. Logistic Regression
# ---------------------------------------------------------------------------

def run_logistic_regression(df, feature_cols, out_dir):
    print("\n── Logistic Regression ────────────────────────────────────────")

    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)

    # Grid search C on inner folds (0–3)
    best_C, best_val_f1 = 1.0, -1.0
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        scores = []
        for fold in range(4):
            X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
            sc = StandardScaler()
            X_tv_s = sc.fit_transform(X_tv)
            X_val_s = sc.transform(X_val)
            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                    solver="lbfgs", random_state=SEED)
            lr.fit(X_tv_s, y_tv)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_val, lr.predict(X_val_s), zero_division=0))
        mean_f1 = float(np.mean(scores))
        print(f"  C={C:<8}  val_F1={mean_f1:.4f}")
        if mean_f1 > best_val_f1:
            best_val_f1, best_C = mean_f1, C

    print(f"  → Best C={best_C}  (val_F1={best_val_f1:.4f})")

    # Final model on full trainval
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    lr = LogisticRegression(C=best_C, class_weight="balanced", max_iter=2000,
                            solver="lbfgs", random_state=SEED)
    lr.fit(X_tr_s, y_tr)

    y_pred = lr.predict(X_te_s)
    y_prob = lr.predict_proba(X_te_s)[:, 1]

    m  = compute_metrics(y_te, y_pred, y_prob)
    ci = bootstrap_ci(y_te, y_pred, y_prob, n_boot=N_BOOT, seed=SEED)
    lat = measure_latency(lambda x: lr.predict(sc.transform(x)), X_te[0])

    # Feature coefficients
    coef = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": lr.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)

    print(f"  TEST  F1={m['f1']:.4f}  prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}  PR-AUC={m.get('pr_auc','N/A')}")
    print(f"  Latency: {lat['latency_mean_ms']:.3f} ms")

    results = {
        "model": "logistic_regression",
        "best_C": best_C,
        "best_val_f1": best_val_f1,
        "test_metrics": m,
        "bootstrap_ci": ci,
        "latency": lat,
    }
    out_dir_lr = out_dir / "logistic_regression"
    out_dir_lr.mkdir(parents=True, exist_ok=True)
    save_pr_curve(y_te, y_prob, "logistic_regression", out_dir_lr)
    save_predictions(df, y_te, y_pred, y_prob, out_dir_lr)
    save_results(results, out_dir_lr, "lr_results")
    joblib.dump({"scaler": sc, "model": lr}, out_dir_lr / "lr_model.pkl")
    coef.to_csv(out_dir_lr / "lr_coefficients.csv", index=False)
    print(f"  Model saved → {out_dir_lr}/lr_model.pkl")
    return results, lr, sc


# ---------------------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------------------

def run_random_forest(df, feature_cols, out_dir):
    print("\n── Random Forest ──────────────────────────────────────────────")

    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)

    param_grid = [
        {"n_estimators": n, "max_depth": d}
        for n in [100, 300]
        for d in [15, 30, None]
    ]

    best_params, best_val_f1 = param_grid[0], -1.0
    for params in param_grid:
        scores = []
        for fold in range(4):
            X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
            rf = RandomForestClassifier(
                **params, class_weight="balanced",
                n_jobs=-1, random_state=SEED,
            )
            rf.fit(X_tv, y_tv)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_val, rf.predict(X_val), zero_division=0))
        mean_f1 = float(np.mean(scores))
        print(f"  n={params['n_estimators']:<4} d={str(params['max_depth']):<6}  val_F1={mean_f1:.4f}")
        if mean_f1 > best_val_f1:
            best_val_f1, best_params = mean_f1, params

    print(f"  → Best: {best_params}  (val_F1={best_val_f1:.4f})")

    # Final model
    rf = RandomForestClassifier(
        **best_params, class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    m  = compute_metrics(y_te, y_pred, y_prob)
    ci = bootstrap_ci(y_te, y_pred, y_prob, n_boot=N_BOOT, seed=SEED)
    lat = measure_latency(rf.predict, X_te[0])

    # Feature importance (MDI)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mdi": rf.feature_importances_,
    }).sort_values("importance_mdi", ascending=False)

    print(f"  TEST  F1={m['f1']:.4f}  prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}  PR-AUC={m.get('pr_auc','N/A')}")
    print(f"  Latency: {lat['latency_mean_ms']:.3f} ms")

    results = {
        "model": "random_forest",
        "best_params": best_params,
        "best_val_f1": best_val_f1,
        "test_metrics": m,
        "bootstrap_ci": ci,
        "latency": lat,
    }
    out_dir_rf = out_dir / "random_forest"
    out_dir_rf.mkdir(parents=True, exist_ok=True)
    save_pr_curve(y_te, y_prob, "random_forest", out_dir_rf)
    save_predictions(df, y_te, y_pred, y_prob, out_dir_rf)
    save_results(results, out_dir_rf, "rf_results")
    joblib.dump(rf, out_dir_rf / "rf_model.pkl")
    imp.to_csv(out_dir_rf / "rf_feature_importance.csv", index=False)
    print(f"  Model saved → {out_dir_rf}/rf_model.pkl")
    return results, rf


# ---------------------------------------------------------------------------
# 4. XGBoost
# ---------------------------------------------------------------------------

def run_xgboost(df, feature_cols, out_dir):
    print("\n── XGBoost ────────────────────────────────────────────────────")
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed — skipping. Install with: pip install xgboost")
        return None, None

    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)
    spw = _pos_weight(y_tr)

    param_grid = [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in [3, 5, 7]
        for lr in [0.05, 0.1]
        for n in [200, 400]
    ]

    best_params, best_val_f1 = param_grid[0], -1.0
    for params in param_grid:
        scores = []
        for fold in range(4):
            X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=spw,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                early_stopping_rounds=20,
                random_state=SEED,
                verbosity=0,
            )
            model.fit(X_tv, y_tv, eval_set=[(X_val, y_val)], verbose=False)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_val, model.predict(X_val), zero_division=0))
        mean_f1 = float(np.mean(scores))
        print(f"  d={params['max_depth']} lr={params['learning_rate']} "
              f"n={params['n_estimators']}  val_F1={mean_f1:.4f}")
        if mean_f1 > best_val_f1:
            best_val_f1, best_params = mean_f1, dict(params)

    print(f"  → Best: {best_params}  (val_F1={best_val_f1:.4f})")

    # Final model — use fold 3 as early-stopping val
    X_tv, y_tv, X_val_es, y_val_es = get_fold_split(df, 3, feature_cols)
    xgb_final = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=SEED,
        verbosity=0,
    )
    # fit on full trainval for final evaluation
    xgb_test = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    )
    xgb_test.fit(X_tr, y_tr)

    y_pred = xgb_test.predict(X_te)
    y_prob = xgb_test.predict_proba(X_te)[:, 1]

    m  = compute_metrics(y_te, y_pred, y_prob)
    ci = bootstrap_ci(y_te, y_pred, y_prob, n_boot=N_BOOT, seed=SEED)
    lat = measure_latency(xgb_test.predict, X_te[0])

    # Feature importance
    scores_gain = xgb_test.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame([
        {"feature": f, "importance_gain": v}
        for f, v in scores_gain.items()
    ]).sort_values("importance_gain", ascending=False)

    print(f"  TEST  F1={m['f1']:.4f}  prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}  PR-AUC={m.get('pr_auc','N/A')}")
    print(f"  Latency: {lat['latency_mean_ms']:.3f} ms")

    results = {
        "model": "xgboost",
        "best_params": best_params,
        "best_val_f1": best_val_f1,
        "test_metrics": m,
        "bootstrap_ci": ci,
        "latency": lat,
    }
    out_dir_xgb = out_dir / "xgboost"
    out_dir_xgb.mkdir(parents=True, exist_ok=True)
    save_pr_curve(y_te, y_prob, "xgboost", out_dir_xgb)
    save_predictions(df, y_te, y_pred, y_prob, out_dir_xgb)
    save_results(results, out_dir_xgb, "xgb_results")
    xgb_test.save_model(str(out_dir_xgb / "xgb_model.ubj"))
    imp.to_csv(out_dir_xgb / "xgb_feature_importance.csv", index=False)
    print(f"  Model saved → {out_dir_xgb}/xgb_model.ubj")
    return results, xgb_test


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(all_results: dict, out_dir: Path) -> pd.DataFrame:
    rows = []
    for name, res in all_results.items():
        if res is None:
            continue
        m  = res.get("test_metrics", res.get("metrics", {}))
        ci = res.get("bootstrap_ci", res.get("ci", {}))
        lat = res.get("latency", {})
        row = {"model": name}
        for k in ("f1", "precision", "recall", "pr_auc", "roc_auc", "mcc"):
            row[k] = m.get(k, None)
            if k in ci:
                row[f"{k}_lo"] = ci[k]["lo"]
                row[f"{k}_hi"] = ci[k]["hi"]
        row["latency_mean_ms"] = lat.get("latency_mean_ms", None)
        rows.append(row)

    cmp = pd.DataFrame(rows)
    cmp_path = out_dir / "comparison_table.csv"
    cmp.to_csv(cmp_path, index=False)
    print(f"\nComparison table → {cmp_path}")
    print(cmp[["model", "f1", "precision", "recall", "pr_auc",
               "mcc", "latency_mean_ms"]].to_string(index=False))
    return cmp


# ---------------------------------------------------------------------------
# McNemar pairwise tests
# ---------------------------------------------------------------------------

def run_mcnemar(df, feature_cols, trained_models: dict, out_dir: Path):
    print("\n── McNemar pairwise tests ─────────────────────────────────────")
    from src.models.evaluate import mcnemar_test

    _, _, X_te, y_te = get_test_split(df, feature_cols)

    preds = {}
    for name, (model, scaler) in trained_models.items():
        if model is None:
            continue
        X = scaler.transform(X_te) if scaler else X_te
        try:
            preds[name] = model.predict(X)
        except Exception:
            pass

    names = list(preds.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            stat, p = mcnemar_test(y_te, preds[a], preds[b])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            rows.append({"model_a": a, "model_b": b, "statistic": stat, "p_value": p, "sig": sig})
            print(f"  {a} vs {b}: p={p:.4f} {sig}")

    pd.DataFrame(rows).to_csv(out_dir / "mcnemar_tests.csv", index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2.1 — Train baseline models")
    p.add_argument("--features-csv", default="data/track_features.csv")
    p.add_argument("--splits-csv",   default="data/splits_folds.csv")
    p.add_argument("--audit-csv",    default="data/data_audit_report.csv")
    p.add_argument("--output-dir",   default="data/results/six_domain_v1")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["majority", "trivial", "lr", "rf", "xgb"],
        default=["trivial", "lr", "rf", "xgb"],
        help="Subset to run. Use 'majority' for only the majority-class baseline.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    features_csv = Path(args.features_csv).expanduser().resolve()
    splits_csv   = Path(args.splits_csv).expanduser().resolve()
    audit_csv    = Path(args.audit_csv).expanduser().resolve()
    out_dir      = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df, feature_cols, _ = load_data(features_csv, splits_csv, audit_csv)

    trivial_res = {}
    lr_res = rf_res = xgb_res = None
    lr_model = lr_sc = rf_model = xgb_model = None

    if "trivial" in args.models:
        trivial_res = run_trivial(df, feature_cols, out_dir)
    elif "majority" in args.models:
        trivial_res = run_majority_baseline(df, feature_cols, out_dir)

    if "lr" in args.models:
        lr_res, lr_model, lr_sc = run_logistic_regression(df, feature_cols, out_dir)
    if "rf" in args.models:
        rf_res, rf_model = run_random_forest(df, feature_cols, out_dir)
    if "xgb" in args.models:
        xgb_res, xgb_model = run_xgboost(df, feature_cols, out_dir)

    all_results = {
        "majority":           trivial_res.get("majority"),
        "random":             trivial_res.get("random_stratified"),
        "single_feature":     trivial_res.get("single_feature"),
        "logistic_regression": lr_res,
        "random_forest":       rf_res,
        "xgboost":             xgb_res,
    }
    build_comparison_table(all_results, out_dir)

    trained_models = {
        name: bundle for name, bundle in {
            "logistic_regression": (lr_model, lr_sc),
            "random_forest":       (rf_model, None),
            "xgboost":             (xgb_model, None),
        }.items()
        if bundle[0] is not None
    }
    if len(trained_models) >= 2:
        run_mcnemar(df, feature_cols, trained_models, out_dir)

    print("\n✓ Step 2.1 complete")


if __name__ == "__main__":
    main()
