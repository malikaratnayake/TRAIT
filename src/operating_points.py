"""
src/operating_points.py
=======================
Phase 3 — XGBoost operating points on the held-out test set (Supp. Table 10).

Reads the test-set predictions saved by src/models/tabular_models.py and
reports precision, recall, F1, the share of insect tracks missed and the
number of false positives at several decision thresholds, including the
deployed threshold chosen on validation data. This shows how the threshold
can be set to trade missed insects against false alarms.

Inputs (under <results-dir>/prauc_protocol/xgboost_prauc/)
-----------------------------------------------------------
  predictions.csv  — filename, y_true, y_pred, score
  results.json     — includes the deployed threshold

Output
------
  operating_points.csv  (same folder)

Usage
-----
  python src/operating_points.py --results-dir data/results/trait_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEFAULT_THRESHOLDS = [0.20, 0.30, 0.40, 0.70]


def operating_points(y_true: np.ndarray, score: np.ndarray, thresholds: list[float],
                     deployed: float) -> pd.DataFrame:
    """Return one row of test metrics per threshold, deployed threshold included."""
    rows = []
    n_insect = int((y_true == 1).sum())
    n_not_insect = int((y_true == 0).sum())
    for thr in sorted(set(thresholds) | {deployed}):
        y_pred = (score >= thr).astype(int)
        missed = int(((y_true == 1) & (y_pred == 0)).sum())
        rows.append({
            "threshold": thr,
            "deployed": bool(np.isclose(thr, deployed)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "missed_insect_pct": 100 * missed / n_insect if n_insect else float("nan"),
            "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
            "n_insect": n_insect,
            "n_not_insect": n_not_insect,
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGBoost operating points on the held-out test set")
    p.add_argument("--results-dir", default="data/results/trait_results",
                   help="Pipeline results directory containing prauc_protocol/xgboost_prauc/")
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                   help=f"Thresholds to evaluate besides the deployed one (default: {DEFAULT_THRESHOLDS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.results_dir).expanduser().resolve() / "prauc_protocol" / "xgboost_prauc"
    preds = pd.read_csv(model_dir / "predictions.csv")
    deployed = float(json.loads((model_dir / "results.json").read_text())["threshold"])

    table = operating_points(preds["y_true"].to_numpy(dtype=int),
                             preds["score"].to_numpy(dtype=float),
                             args.thresholds, deployed)
    out_path = model_dir / "operating_points.csv"
    table.round({"precision": 4, "recall": 4, "f1": 4, "missed_insect_pct": 2}).to_csv(out_path, index=False)

    print(f"\n  {'τ':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Missed':>7}  {'FP':>5}")
    for _, r in table.iterrows():
        tag = "  (deployed)" if r["deployed"] else ""
        print(f"  {r['threshold']:>6.3f}  {r['precision']:>6.3f}  {r['recall']:>6.3f}  "
              f"{r['f1']:>6.3f}  {r['missed_insect_pct']:>6.1f}%  {r['false_positives']:>5d}{tag}")
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
