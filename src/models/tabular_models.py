"""
src/models/tabular_models.py
============================
Phase 3 — Logistic Regression, Random Forest and XGBoost on the 125 features.

All three models follow the same protocol (paper Sec. 4.1, Supp. Table 8):

  - hyperparameters chosen by mean validation PR-AUC over folds 0–3
    (XGBoost also early-stops on validation PR-AUC, 400-round cap)
  - decision threshold chosen by maximising F1 on out-of-fold predictions
  - final model refit on folds 0–3 and evaluated once on the held-out test
    fold (fold 4): precision, recall, F1, PR-AUC with bootstrap CIs, latency
  - class imbalance: balanced class weights (LR, RF) or scale_pos_weight set
    from the training class ratio (XGBoost)

Outputs (under <output-dir>/prauc_protocol/)
--------------------------------------------
  prauc_protocol_comparison.csv          — one row per model (Table 3 rows)
  <model>_prauc/results.json             — selected hyperparameters, threshold,
                                           test metrics, CIs, latency
  <model>_prauc/predictions.csv          — per-track test predictions
  <model>_prauc/<model>_prauc_pr_curve.csv
  <model>_prauc/model.pkl | model.ubj    — fitted model

Usage
-----
  python src/models/tabular_models.py \\
      --features-csv data/results/trait_results/track_features.csv \\
      --splits-csv   data/splits_folds.csv \\
      --output-dir   data/results/trait_results
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.data_loader import load_data, get_fold_split, get_test_split
from src.models.evaluate import compute_metrics, bootstrap_ci, measure_latency, save_results
from src.models.majority_baseline import save_pr_curve

SEED = 42
N_BOOT = 1000


def _pos_weight(y: np.ndarray) -> float:
    """Negative-to-positive class ratio, used as XGBoost's scale_pos_weight."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def _find_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return the threshold in [0.05, 0.95] (step 0.01) that maximises F1, and that F1."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (y_score >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_thr, best_f1 = float(thr), float(score)
    return round(best_thr, 3), round(best_f1, 4)


def _test_filenames(df: pd.DataFrame) -> list[str]:
    """Filenames of the held-out test tracks, in the order get_test_split returns them."""
    return list(df.loc[df["split_role"] == "test", "filename"])


def _save_variant(
    *,
    out_dir: Path,
    slug: str,
    display_name: str,
    n_features: int | str,
    feature_set: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    threshold_val_f1: float,
    latency: dict,
    extra: dict,
    filenames: list[str],
) -> dict:
    """Score one fitted model on the test set and save its outputs.

    Writes results.json, predictions.csv and the PR-curve CSV to
    out_dir/<slug>/ and returns the summary row for the comparison table.
    """
    y_pred = (y_score >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_score)
    ci = bootstrap_ci(y_true, y_pred, y_score, n_boot=N_BOOT, seed=SEED)

    result = {
        "model": slug,
        "display_name": display_name,
        "feature_set": feature_set,
        "n_features": n_features,
        "selection_protocol": {
            "model_selection_metric": "validation_pr_auc",
            "early_stopping_metric": "validation_pr_auc_where_applicable",
            "threshold_metric": "validation_f1",
            "primary_test_metric": "heldout_test_pr_auc",
        },
        "threshold": threshold,
        "threshold_val_f1": threshold_val_f1,
        "test_metrics": metrics,
        "bootstrap_ci": ci,
        "latency": latency,
        **extra,
    }

    model_dir = out_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    save_results(result, model_dir, "results")
    save_pr_curve(y_true, y_score, slug, model_dir)
    pd.DataFrame({
        "filename": filenames,
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
        "score": y_score,
    }).to_csv(model_dir / "predictions.csv", index=False)

    f1_ci = ci.get("f1", {})
    pr_ci = ci.get("pr_auc", {})
    return {
        "model": display_name,
        "slug": slug,
        "n_features": n_features,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "f1_ci_lo": f1_ci.get("lo"),
        "f1_ci_hi": f1_ci.get("hi"),
        "pr_auc": metrics.get("pr_auc"),
        "pr_auc_ci_lo": pr_ci.get("lo"),
        "pr_auc_ci_hi": pr_ci.get("hi"),
        "latency_mean_ms": latency.get("latency_mean_ms"),
        "threshold": threshold,
        **extra,
    }


def _fit_lr(X: np.ndarray, y: np.ndarray, C: float):
    """Fit a standardised, class-balanced L2 logistic regression with inverse regularisation C."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=SEED,
    )
    model.fit(Xs, y)
    return scaler, model


def _score_lr(scaler, model, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities from a fitted scaler + logistic regression."""
    return model.predict_proba(scaler.transform(X))[:, 1]


def run_lr(df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> dict:
    """Select C by validation PR-AUC, tune the threshold, refit and evaluate on the test set."""
    print("\n── Logistic Regression — PR-AUC selection ─────────────────────")
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0]

    best_C, best_ap = Cs[0], -1.0
    for C in Cs:
        scores = []
        for fold in range(4):
            X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
            scaler, model = _fit_lr(X_tv, y_tv, C)
            scores.append(average_precision_score(y_val, _score_lr(scaler, model, X_val)))
        mean_ap = float(np.mean(scores))
        print(f"  C={C:<8}  val_PR_AUC={mean_ap:.4f}")
        if mean_ap > best_ap:
            best_C, best_ap = C, mean_ap

    oof_scores, oof_true = [], []
    for fold in range(4):
        X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
        scaler, model = _fit_lr(X_tv, y_tv, best_C)
        oof_scores.append(_score_lr(scaler, model, X_val))
        oof_true.append(y_val)
    threshold, thr_f1 = _find_f1_threshold(np.concatenate(oof_true), np.concatenate(oof_scores))

    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)
    scaler, model = _fit_lr(X_tr, y_tr, best_C)
    y_score = _score_lr(scaler, model, X_te)
    latency = measure_latency(
        lambda x: (_score_lr(scaler, model, x) >= threshold).astype(int),
        X_te[0],
    )
    model_dir = out_dir / "logistic_regression_prauc"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, model_dir / "model.pkl")
    return _save_variant(
        out_dir=out_dir,
        slug="logistic_regression_prauc",
        display_name="Logistic Regression (PR-AUC select)",
        n_features=len(feature_cols),
        feature_set="engineered_125",
        y_true=y_te,
        y_score=y_score,
        threshold=threshold,
        threshold_val_f1=thr_f1,
        latency=latency,
        filenames=_test_filenames(df),
        extra={"best_C": best_C, "best_val_pr_auc": round(best_ap, 4)},
    )


def _fit_rf(X: np.ndarray, y: np.ndarray, params: dict, n_jobs: int):
    """Fit a class-balanced random forest with the given n_estimators / max_depth."""
    model = RandomForestClassifier(
        **params,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=SEED,
    )
    model.fit(X, y)
    return model


def run_rf(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    n_jobs: int,
) -> dict:
    """Grid-search the random forest by validation PR-AUC, tune the threshold and evaluate on the test set."""
    print("\n── Random Forest — PR-AUC selection ───────────────────────────")
    param_grid = [
        {"n_estimators": n, "max_depth": d}
        for n in [100, 300]
        for d in [15, 30, None]
    ]

    best_params, best_ap = param_grid[0], -1.0
    for params in param_grid:
        scores = []
        for fold in range(4):
            X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
            model = _fit_rf(X_tv, y_tv, params, n_jobs)
            scores.append(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
        mean_ap = float(np.mean(scores))
        print(f"  n={params['n_estimators']:<4} d={str(params['max_depth']):<6}  val_PR_AUC={mean_ap:.4f}")
        if mean_ap > best_ap:
            best_params, best_ap = dict(params), mean_ap

    oof_scores, oof_true = [], []
    for fold in range(4):
        X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
        model = _fit_rf(X_tv, y_tv, best_params, n_jobs)
        oof_scores.append(model.predict_proba(X_val)[:, 1])
        oof_true.append(y_val)
    threshold, thr_f1 = _find_f1_threshold(np.concatenate(oof_true), np.concatenate(oof_scores))

    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)
    model = _fit_rf(X_tr, y_tr, best_params, n_jobs)
    y_score = model.predict_proba(X_te)[:, 1]
    latency = measure_latency(
        lambda x: (model.predict_proba(x)[:, 1] >= threshold).astype(int),
        X_te[0],
    )
    model_dir = out_dir / "random_forest_prauc"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")
    return _save_variant(
        out_dir=out_dir,
        slug="random_forest_prauc",
        display_name="Random Forest (PR-AUC select)",
        n_features=len(feature_cols),
        feature_set="engineered_125",
        y_true=y_te,
        y_score=y_score,
        threshold=threshold,
        threshold_val_f1=thr_f1,
        latency=latency,
        filenames=_test_filenames(df),
        extra={
            "best_params": best_params,
            "best_val_pr_auc": round(best_ap, 4),
            "n_jobs": n_jobs,
        },
    )


def _fit_xgb(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    eval_set=None,
    early_stop: bool = False,
    n_jobs: int = -1,
):
    """Fit an XGBoost classifier.

    scale_pos_weight is set from the training class ratio; with early_stop,
    training stops after 20 rounds without improvement in validation PR-AUC.
    """
    import xgboost as xgb

    kwargs = {
        **params,
        "scale_pos_weight": _pos_weight(y),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "random_state": SEED,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    if early_stop:
        kwargs["early_stopping_rounds"] = 20
    model = xgb.XGBClassifier(**kwargs)
    if eval_set is None:
        model.fit(X, y, verbose=False)
    else:
        model.fit(X, y, eval_set=[eval_set], verbose=False)
    return model


def _best_n_estimators(model, fallback: int) -> int:
    """Number of boosting rounds kept by early stopping (or the fallback if none)."""
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return int(fallback)
    return max(1, int(best_iteration) + 1)


def _run_xgb_protocol(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    *,
    slug: str,
    display_name: str,
    feature_set: str,
    n_jobs: int,
) -> dict:
    """Grid-search XGBoost by validation PR-AUC, then refit and evaluate on the test set.

    The number of boosting rounds for the final model is the median of the
    early-stopping rounds across folds 0–3; the threshold maximises
    out-of-fold F1.
    """
    print(f"\n── {display_name} ─────────────────────────────────")
    param_grid = [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in [3, 5, 7]
        for lr in [0.05, 0.1]
        for n in [200, 400]
    ]

    best_params, best_ap, best_rounds = param_grid[0], -1.0, []
    for params in param_grid:
        scores, rounds = [], []
        for fold in range(4):
            X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
            model = _fit_xgb(
                X_tv,
                y_tv,
                params,
                eval_set=(X_val, y_val),
                early_stop=True,
                n_jobs=n_jobs,
            )
            scores.append(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
            rounds.append(_best_n_estimators(model, params["n_estimators"]))
        mean_ap = float(np.mean(scores))
        print(f"  d={params['max_depth']} lr={params['learning_rate']} "
              f"n={params['n_estimators']}  val_PR_AUC={mean_ap:.4f}")
        if mean_ap > best_ap:
            best_params, best_ap, best_rounds = dict(params), mean_ap, rounds

    selected_n = int(np.median(best_rounds)) if best_rounds else int(best_params["n_estimators"])

    oof_scores, oof_true, oof_rounds = [], [], []
    for fold in range(4):
        X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
        model = _fit_xgb(
            X_tv,
            y_tv,
            best_params,
            eval_set=(X_val, y_val),
            early_stop=True,
            n_jobs=n_jobs,
        )
        oof_scores.append(model.predict_proba(X_val)[:, 1])
        oof_true.append(y_val)
        oof_rounds.append(_best_n_estimators(model, best_params["n_estimators"]))
    if oof_rounds:
        selected_n = int(np.median(oof_rounds))
    threshold, thr_f1 = _find_f1_threshold(np.concatenate(oof_true), np.concatenate(oof_scores))

    final_params = dict(best_params)
    final_params["n_estimators"] = selected_n
    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)
    model = _fit_xgb(X_tr, y_tr, final_params, n_jobs=n_jobs)
    y_score = model.predict_proba(X_te)[:, 1]
    latency = measure_latency(
        lambda x: (model.predict_proba(x)[:, 1] >= threshold).astype(int),
        X_te[0],
    )
    model_dir = out_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.ubj"))

    return _save_variant(
        out_dir=out_dir,
        slug=slug,
        display_name=display_name,
        n_features=len(feature_cols),
        feature_set=feature_set,
        y_true=y_te,
        y_score=y_score,
        threshold=threshold,
        threshold_val_f1=thr_f1,
        latency=latency,
        filenames=_test_filenames(df),
        extra={
            "best_params": best_params,
            "selected_n_estimators": selected_n,
            "best_val_pr_auc": round(best_ap, 4),
            "n_jobs": n_jobs,
        },
    )


def run_xgb(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    n_jobs: int,
) -> dict:
    """Run the XGBoost protocol on the 125-feature set."""
    return _run_xgb_protocol(
        df,
        feature_cols,
        out_dir,
        slug="xgboost_prauc",
        display_name="XGBoost (PR-AUC select)",
        feature_set="engineered_125",
        n_jobs=n_jobs,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Phase 3 — Logistic Regression, Random Forest and XGBoost")
    parser.add_argument("--features-csv", default="data/results/trait_results/track_features.csv")
    parser.add_argument("--splits-csv", default="data/splits_folds.csv")
    parser.add_argument("--output-dir", default="data/results/trait_results")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("TRAIT_N_JOBS", "-1")),
        help="Parallel jobs for Random Forest and XGBoost. Use 1 for final reproducible runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lr", "rf", "xgb"],
        default=["lr", "rf", "xgb"],
        help="Subset of models to train (default: all three)",
    )
    return parser.parse_args()


def main() -> None:
    """Train the requested models and write prauc_protocol_comparison.csv."""
    args = parse_args()
    features_csv = Path(args.features_csv).expanduser().resolve()
    splits_csv = Path(args.splits_csv).expanduser().resolve()
    root_out = Path(args.output_dir).expanduser().resolve()
    out_dir = root_out / "prauc_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    df, feature_cols, _ = load_data(features_csv, splits_csv)

    rows = []
    if "lr" in args.models:
        rows.append(run_lr(df, feature_cols, out_dir))
    if "rf" in args.models:
        rows.append(run_rf(df, feature_cols, out_dir, args.n_jobs))
    if "xgb" in args.models:
        if importlib.util.find_spec("xgboost") is None:
            print(
                "\nWARNING: xgboost is not installed in this Python environment; "
                "skipping XGBoost. Install with: python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
        else:
            rows.append(run_xgb(df, feature_cols, out_dir, args.n_jobs))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "prauc_protocol_comparison.csv", index=False)
    print(f"\nPR-AUC protocol comparison -> {out_dir / 'prauc_protocol_comparison.csv'}")
    if comparison.empty:
        print("No requested tabular models were trained.")
    else:
        print(comparison[["model", "f1", "pr_auc", "threshold"]].to_string(index=False))


if __name__ == "__main__":
    main()
