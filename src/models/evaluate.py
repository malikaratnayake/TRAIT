"""
src/models/evaluate.py
======================
Evaluation utilities shared by the classifiers in Phases 3–5: test metrics
(precision, recall, F1, MCC, PR-AUC, ROC-AUC), 95% bootstrap confidence
intervals over test-set resamples, and single-sample inference latency
(mean over 500 timed calls after 50 warm-up calls).

Public API
----------
  compute_metrics(y_true, y_pred, y_prob)  -> dict
  bootstrap_ci(y_true, y_pred, y_prob, n_boot, seed)  -> dict of (mean, lo, hi)
  measure_latency(predict_fn, X_sample, n_reps)  -> dict
  save_results(results, out_dir, stem)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Return a dict of evaluation metrics.

    Parameters
    ----------
    y_true : binary array  (0 = not_insect, 1 = insect)
    y_pred : binary predictions
    y_prob : predicted probability for the positive class (optional)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pr  = precision_score(y_true, y_pred, zero_division=0)
    re  = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    out = dict(
        f1=round(float(f1), 4),
        precision=round(float(pr), 4),
        recall=round(float(re), 4),
        mcc=round(float(mcc), 4),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        n_total=int(len(y_true)),
        n_positive=int(y_true.sum()),
    )
    if y_prob is not None:
        pr_auc  = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        out["pr_auc"]  = round(float(pr_auc), 4)
        out["roc_auc"] = round(float(roc_auc), 4)
    return out


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Return 95% bootstrap CIs for all metrics.

    Returns dict:  {metric: {"mean": float, "lo": float, "hi": float}}
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    records = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        yb = y_prob[idx] if y_prob is not None else None
        # skip degenerate bootstrap samples (only one class)
        if len(np.unique(yt)) < 2:
            continue
        records.append(compute_metrics(yt, yp, yb))

    df = pd.DataFrame(records)
    result = {}
    for col in df.columns:
        vals = df[col].dropna().values
        result[col] = {
            "mean": round(float(vals.mean()), 4),
            "lo":   round(float(np.percentile(vals, 2.5)), 4),
            "hi":   round(float(np.percentile(vals, 97.5)), 4),
        }
    return result


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(
    predict_fn: Callable,
    X_sample: np.ndarray,
    n_reps: int = 500,
    warmup: int = 50,
) -> dict:
    """Measure single-sample inference latency in milliseconds.

    Parameters
    ----------
    predict_fn : callable that accepts a (1, n_features) array
    X_sample   : one row of features, shape (n_features,) or (1, n_features)
    n_reps     : number of timed repetitions
    warmup     : untimed warm-up repetitions
    """
    x = X_sample.reshape(1, -1) if X_sample.ndim == 1 else X_sample[:1]
    # warm up
    for _ in range(warmup):
        predict_fn(x)
    # timed reps
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        predict_fn(x)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    times = np.array(times)
    return {
        "latency_mean_ms": round(float(times.mean()), 4),
        "latency_std_ms":  round(float(times.std()), 4),
        "latency_min_ms":  round(float(times.min()), 4),
        "latency_p95_ms":  round(float(np.percentile(times, 95)), 4),
        "latency_max_ms":  round(float(times.max()), 4),
    }


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_results(results: dict, out_dir: Path, stem: str) -> None:
    """Save a results dict as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {path.name}")
