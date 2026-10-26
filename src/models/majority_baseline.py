"""
src/models/majority_baseline.py
===============================
Phase 3 — Majority-class baseline.

Always predicts the most frequent training class (not_insect) and is reported
as the prevalence reference in Table 3. Trained on folds 0–3 and evaluated on
the held-out test fold (fold 4) with bootstrap confidence intervals and
single-sample latency.

Also provides save_pr_curve() and save_predictions(), which
src/models/tabular_models.py reuses.

Outputs (under <output-dir>/trivial/)
-------------------------------------
  trivial_results.json       — test metrics, bootstrap CIs and latency
  majority_pr_curve.csv      — precision-recall curve points
  majority_predictions.csv   — per-track test predictions

Usage
-----
  python src/models/majority_baseline.py \\
      --features-csv data/results/trait_results/track_features.csv \\
      --splits-csv   data/splits_folds.csv \\
      --output-dir   data/results/trait_results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.data_loader import load_data, get_test_split
from src.models.evaluate import compute_metrics, bootstrap_ci, measure_latency, save_results

SEED = 42
N_BOOT = 1000


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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line options."""
    p = argparse.ArgumentParser(description="Phase 3 — Majority-class baseline")
    p.add_argument("--features-csv", default="data/results/trait_results/track_features.csv")
    p.add_argument("--splits-csv",   default="data/splits_folds.csv")
    p.add_argument("--output-dir",   default="data/results/trait_results")
    return p.parse_args()


def main():
    """Load the data and run the majority baseline."""
    args = parse_args()
    features_csv = Path(args.features_csv).expanduser().resolve()
    splits_csv   = Path(args.splits_csv).expanduser().resolve()
    out_dir      = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df, feature_cols, _ = load_data(features_csv, splits_csv)
    run_majority_baseline(df, feature_cols, out_dir)
    print("\n✓ Majority baseline complete")


if __name__ == "__main__":
    main()
