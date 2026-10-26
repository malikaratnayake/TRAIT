"""
src/train_rocket.py — MiniROCKET baseline (Phase 3)
==================================================
Runs MiniROCKET on the 28k insect-track dataset
using the same held-out test protocol as all other models:
  - Train : folds 0, 1, 2
  - Val   : fold 3  (threshold sweep)
  - Test  : fold 4  (held-out, never touched during development)

Input :  data/tracks/          — per-track CSVs with columns [nframe, x, y]
         data/splits_folds.csv — fold assignments
Output:  data/results/six_domain_v1/rocket/rocket_results.csv
         data/results/six_domain_v1/rocket/rocket_predictions.csv

Run from the TRAIT project root (aeon installed in trait-env, see requirements.txt):
    python src/train_rocket.py [--max_len 512]

Dependencies:
    aeon == 1.3.0, numba < 0.62, numpy < 2, scikit-learn, scipy, pandas

Preprocessing parity (2026-08 fix): gap-filling now reuses the exact
linear-interpolation-then-SG-smoothing logic used by src/generate_features.py
(via src/utils/smoothing.smooth_coords) instead of dropping missing-detection
frames before smoothing. See train_fcn.py for the same fix and rationale.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# aeon/numba MiniROCKET segfaults on this macOS/ARM environment when numba uses
# its OpenMP backend with more than one worker. The workqueue backend preserves
# parallel execution without hitting that native crash path.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.smoothing import smooth_coords  # same fn used by generate_features.py

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
SG_WINDOW = 11
SG_POLY   = 2
N_KERNELS = 10_000       # MiniROCKET default
CHANNELS  = ["x_norm", "y_norm", "speed_norm", "dheading"]  # 4 input channels (same as neural)
SEED      = 42
# Median track ~150 frames (p95 ≈ 503); 128 truncated the majority of tracks.
# 512 keeps ~95% of tracks intact while matching train_fcn.py's window.
MAX_LEN_DEFAULT = 512
DEFAULT_N_JOBS = int(os.environ.get("TRAIT_N_JOBS", os.environ.get("ROCKET_N_JOBS", "-1")))

# Paths (relative to project root)
TRACKS_DIR     = Path("data/tracks")
SPLITS_CSV     = Path("data/splits_folds.csv")
RESULTS_DIR    = Path("data/results/six_domain_v1/rocket")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def interpolate_and_smooth(x_raw: np.ndarray, y_raw: np.ndarray):
    """Gap-fill via linear interpolation, then SG-smooth.

    Reuses smooth_coords (src/utils/smoothing.py) with the exact same
    interpolation step used by src/generate_features.py's on-the-fly
    smoothing path (main(), ~line 2213): NaN gaps are linearly interpolated
    over row index before the SG filter is applied, so this baseline
    consumes the identical gap-filled trajectory as the engineered features,
    rather than a version with missing-detection frames dropped.
    """
    x = x_raw.astype(float).copy()
    y = y_raw.astype(float).copy()
    idx = np.arange(len(x))
    for arr in (x, y):
        nans = np.isnan(arr)
        if nans.any() and (~nans).any():
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    sx, sy = smooth_coords(x, y, window_length=SG_WINDOW, polyorder=SG_POLY)
    return sx.astype(np.float32), sy.astype(np.float32)


def normalise_track_frame(sx: np.ndarray, sy: np.ndarray, speed: np.ndarray):
    """Remove absolute frame position and scale each track by its robust spread."""
    sx_centered = sx - np.nanmean(sx)
    sy_centered = sy - np.nanmean(sy)
    radial = np.sqrt(sx_centered**2 + sy_centered**2)
    scale = float(np.nanpercentile(radial, 90))
    if not np.isfinite(scale) or scale < 1.0:
        scale = 1.0
    return sx_centered / scale, sy_centered / scale, speed / scale


def load_track(csv_path: Path, max_len: int):   # -> np.ndarray | None  (Python 3.10+)
    """
    Load a track CSV, apply SG smoothing, compute derived channels, and
    return a (C, T) array padded / truncated to max_len.

    Returns None if the file is missing or has fewer than 3 valid (non-NaN)
    detections. Gaps between valid detections are linearly interpolated
    (matching the engineered-feature pipeline) rather than dropped.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    n_valid = df[["x", "y"]].dropna().shape[0]
    if n_valid < 3:
        return None

    x_raw = df["x"].to_numpy(dtype=np.float64)
    y_raw = df["y"].to_numpy(dtype=np.float64)

    sx, sy = interpolate_and_smooth(x_raw, y_raw)

    # Speed (pixels/frame)
    dx = np.diff(sx, prepend=sx[0])
    dy = np.diff(sy, prepend=sy[0])
    speed = np.sqrt(dx**2 + dy**2)

    # Heading change (radians, signed)
    heading = np.arctan2(dy, dx)
    dheading = np.diff(heading, prepend=heading[0])
    # Wrap to [-π, π]
    dheading = (dheading + np.pi) % (2 * np.pi) - np.pi

    sx_norm, sy_norm, speed_norm = normalise_track_frame(sx, sy, speed)

    channels = np.stack([sx_norm, sy_norm, speed_norm, dheading], axis=0)  # (4, T)

    # Truncate or pad to max_len
    T = channels.shape[1]
    if T >= max_len:
        channels = channels[:, :max_len]
    else:
        pad = np.zeros((channels.shape[0], max_len - T), dtype=np.float32)
        channels = np.concatenate([channels, pad], axis=1)

    return channels.astype(np.float32)


def metric_ci(y_true, y_pred, y_score, n_boot=1000, rng=SEED):
    """Bootstrap 95% CIs for F1 and PR-AUC."""
    rng = np.random.default_rng(rng)
    idx = np.arange(len(y_true))
    f1_scores = []
    pr_auc_scores = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        f1_scores.append(f1_score(y_true[s], y_pred[s], zero_division=0))
        pr_auc_scores.append(average_precision_score(y_true[s], y_score[s]))

    def bounds(values):
        values = np.sort(np.asarray(values, dtype=float))
        if len(values) == 0:
            return np.nan, np.nan
        return np.percentile(values, 2.5), np.percentile(values, 97.5)

    f1_lo, f1_hi = bounds(f1_scores)
    pr_lo, pr_hi = bounds(pr_auc_scores)
    return {
        "f1_lo": f1_lo,
        "f1_hi": f1_hi,
        "pr_auc_lo": pr_lo,
        "pr_auc_hi": pr_hi,
    }


def measure_latency(predict_fn, X_sample: np.ndarray, n_reps: int = 200, warmup: int = 20) -> dict:
    """Measure single-sample inference latency in milliseconds."""
    x = X_sample[None, ...] if X_sample.ndim == 2 else X_sample[:1]
    for _ in range(warmup):
        predict_fn(x)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        predict_fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times, dtype=float)
    return {
        "latency_mean_ms": round(float(times.mean()), 4),
        "latency_std_ms":  round(float(times.std()), 4),
        "latency_min_ms":  round(float(times.min()), 4),
        "latency_p95_ms":  round(float(np.percentile(times, 95)), 4),
        "latency_max_ms":  round(float(times.max()), 4),
    }


def save_pr_curve(y_true, y_score, model_name: str) -> Path:
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
    path = RESULTS_DIR / f"{model_name.lower()}_pr_curve.csv"
    curve.to_csv(path, index=False)
    return path


def resolve_rocket_n_jobs(n_jobs: int) -> int:
    """Resolve sklearn-style n_jobs to an explicit numba/aeon thread count."""
    requested = int(n_jobs)
    if requested == 0:
        raise ValueError("n_jobs=0 is invalid; use -1 for all available threads or a positive integer.")

    try:
        from numba import config, get_num_threads

        thread_cap = int(getattr(config, "NUMBA_NUM_THREADS", 0) or get_num_threads())
    except Exception:
        thread_cap = int(os.cpu_count() or 1)
    thread_cap = max(1, thread_cap)

    if requested < 0:
        # Match joblib semantics: -1 uses all available threads, -2 leaves one free, etc.
        resolved = thread_cap + 1 + requested
    else:
        resolved = requested

    return max(1, min(int(resolved), thread_cap))


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_minirocket():
    """Import aeon's MiniRocket transformer."""
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket

        return MiniRocket
    except ImportError as exc:
        print(f"ERROR: unable to import MiniROCKET from aeon: {exc}")
        print(f"Reinstall with: {sys.executable} -m pip install -r requirements.txt")
        raise


def check_dependencies():
    """Verify the transformer import without starting the experiment."""
    load_minirocket()
    print(f"MiniROCKET dependency import OK: {sys.executable}")


def run(max_len: int = MAX_LEN_DEFAULT, n_jobs: int = DEFAULT_N_JOBS,
        model_name: str | None = None, limit: int | None = None):
    model_name = model_name or "MiniROCKET"
    selection_metric = "pr_auc"
    requested_n_jobs = int(n_jobs)

    # ── 1. Import aeon ──────────────────────────────────────────────────────
    transformer_class = load_minirocket()
    n_jobs = resolve_rocket_n_jobs(requested_n_jobs)

    print(f"\n{'='*60}")
    print(f" {model_name}  |  max_len={max_len}  |  channels={CHANNELS}  |  n_jobs={n_jobs}")
    if n_jobs != requested_n_jobs:
        print(f" Requested n_jobs={requested_n_jobs}; resolved to {n_jobs} for numba/aeon.")
    print(f" Selection metric: {selection_metric}")
    print(f"{'='*60}\n")

    # ── 2. Load splits ──────────────────────────────────────────────────────
    print("Loading splits…")
    splits = pd.read_csv(SPLITS_CSV)
    # Expected columns: fold, filename, (split_role optional)
    # fold 4 = test; folds 0-2 = train; fold 3 = val
    train_files = set(splits[splits.fold.isin([0, 1, 2])]["filename"])
    val_files   = set(splits[splits.fold == 3]["filename"])
    test_files  = set(splits[splits.fold == 4]["filename"])

    # Ground truth labels
    try:
        labels_df = pd.read_csv("data/track_classification.csv")
        # Support "filepath" column (e.g. "./tracks/FarmA_...csv") as well as "filename"
        if "filename" not in labels_df.columns and "filepath" in labels_df.columns:
            labels_df["filename"] = labels_df["filepath"].apply(lambda p: Path(p).name)
    except FileNotFoundError:
        # Fall back: derive from splits if it has a label column
        labels_df = splits[["filename"]].copy()
        if "label" in splits.columns:
            labels_df["label"] = splits["label"]
        else:
            raise FileNotFoundError(
                "Cannot find data/track_classification.csv — "
                "ensure it has columns [filename/filepath, label] where label ∈ {insect, not_insect}"
            )

    label_map = dict(zip(labels_df["filename"], labels_df["label"]))
    # Convert to binary
    binary = {f: 1 if str(v).lower() in ("insect", "1") else 0
              for f, v in label_map.items()}

    # ── 3. Load and featurise tracks ────────────────────────────────────────
    def build_array(file_set: set, desc: str):
        files_sorted = sorted(file_set)
        if limit is not None:
            files_sorted = files_sorted[:limit]
        X, y, fnames = [], [], []
        missing = 0
        for fn in files_sorted:
            path = TRACKS_DIR / fn
            arr = load_track(path, max_len)
            if arr is None:
                missing += 1
                continue
            lbl = binary.get(fn)
            if lbl is None:
                missing += 1
                continue
            X.append(arr)
            y.append(lbl)
            fnames.append(fn)
        X = np.stack(X, axis=0)  # (N, C, T)
        y = np.array(y, dtype=int)
        print(f"  {desc}: {len(X)} tracks  (skipped {missing})  "
              f"  insect={y.mean():.1%}")
        return X, y, fnames

    print("Loading tracks (this may take ~1-2 min for 28k tracks)…")
    t0 = time.time()
    X_train, y_train, _ = build_array(train_files, "train")
    X_val,   y_val,   _ = build_array(val_files,   "val  ")
    X_test,  y_test,  fnames_test = build_array(test_files, "test ")
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # ── 4. Fit ROCKET transform ─────────────────────────────────────────────
    print(f"Fitting {model_name} transform on train set…")
    t0 = time.time()
    rocket = transformer_class(n_kernels=N_KERNELS, random_state=SEED, n_jobs=n_jobs)
    rocket.fit(X_train)
    print(f"  Fit done in {time.time()-t0:.1f}s")

    print("Transforming sets…")
    t0 = time.time()
    Xf_train = rocket.transform(X_train)
    Xf_val   = rocket.transform(X_val)
    Xf_test  = rocket.transform(X_test)
    print(f"  Transform done in {time.time()-t0:.1f}s  "
          f"  feature_dim={Xf_train.shape[1]:,}")

    # Normalise (recommended for Ridge)
    scaler = StandardScaler(with_mean=False)
    Xf_train = scaler.fit_transform(Xf_train)
    Xf_val   = scaler.transform(Xf_val)
    Xf_test  = scaler.transform(Xf_test)

    # Combine train + val for final model
    Xf_trainval = np.vstack([Xf_train, Xf_val])
    y_trainval  = np.concatenate([y_train, y_val])

    # ── 5. Ridge classifier (standard for ROCKET) ───────────────────────────
    print("\nFitting Ridge classifier on train (for threshold sweep)…")
    alphas = np.logspace(-4, 4, 20)
    selected_alpha = None
    best_ap = -1.0
    for alpha in alphas:
        candidate = RidgeClassifier(alpha=float(alpha), class_weight="balanced")
        candidate.fit(Xf_train, y_train)
        score = average_precision_score(y_val, candidate.decision_function(Xf_val))
        if score > best_ap:
            best_ap, selected_alpha = float(score), float(alpha)
    selected_alpha_pr_auc = best_ap
    ridge_inner = RidgeClassifier(alpha=selected_alpha, class_weight="balanced")
    print(f"  Selected alpha={selected_alpha:.6g}  (val PR-AUC={best_ap:.4f})")
    ridge_inner.fit(Xf_train, y_train)

    # ROCKET's decision_function is signed distance → use for threshold sweep
    val_scores = ridge_inner.decision_function(Xf_val)

    print("Sweeping threshold on val fold…")
    best_thr, best_f1 = 0.0, 0.0
    for thr in np.linspace(val_scores.min(), val_scores.max(), 200):
        pred = (val_scores >= thr).astype(int)
        f = f1_score(y_val, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr
    print(f"  Val-optimal threshold: {best_thr:.4f}  (val F1 = {best_f1:.4f})")

    # ── 6. Retrain on train+val, evaluate on test ───────────────────────────
    print("\nRetraining on train+val for final evaluation…")
    ridge_final = RidgeClassifier(alpha=selected_alpha, class_weight="balanced")
    ridge_final.fit(Xf_trainval, y_trainval)

    test_scores = ridge_final.decision_function(Xf_test)
    y_pred = (test_scores >= best_thr).astype(int)

    def predict_one_track(x):
        x_feat = rocket.transform(x)
        x_feat = scaler.transform(x_feat)
        return ridge_final.predict(x_feat)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prauc = average_precision_score(y_test, test_scores)
    ci = metric_ci(y_test, y_pred, test_scores)
    ci_lo, ci_hi = ci["f1_lo"], ci["f1_hi"]
    latency = measure_latency(predict_one_track, X_test[0])

    print(f"\n{'─'*40}")
    print(f"  Model      : {model_name}")
    print(f"  max_len    : {max_len}")
    print(f"  Threshold  : {best_thr:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Test F1    : {f1:.4f}  95% CI [{ci_lo:.3f}–{ci_hi:.3f}]")
    print(f"  PR-AUC     : {prauc:.4f}")
    print(f"  Latency    : {latency['latency_mean_ms']:.3f} ms/sample")
    print(f"{'─'*40}\n")

    # ── 7. Save outputs ─────────────────────────────────────────────────────
    results = pd.DataFrame([{
        "model":     model_name,
        "max_len":   max_len,
        "channels":  ",".join(CHANNELS),
        "n_jobs_requested": requested_n_jobs,
        "n_jobs_resolved": n_jobs,
        "selection_metric": selection_metric,
        "selected_alpha": selected_alpha,
        "selected_alpha_val_pr_auc": selected_alpha_pr_auc,
        "threshold": round(best_thr, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "f1_lo":     round(ci_lo, 3),
        "f1_hi":     round(ci_hi, 3),
        "ci_lo":     round(ci_lo, 3),
        "ci_hi":     round(ci_hi, 3),
        "pr_auc":    round(prauc, 4),
        "pr_auc_lo": round(ci["pr_auc_lo"], 4),
        "pr_auc_hi": round(ci["pr_auc_hi"], 4),
        **latency,
    }])
    results_path = RESULTS_DIR / "rocket_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")

    preds = pd.DataFrame({
        "filename": fnames_test,
        "y_true":   y_test,
        "y_pred":   y_pred,
        "score":    test_scores,
    })
    preds_path = RESULTS_DIR / "rocket_predictions.csv"
    preds.to_csv(preds_path, index=False)
    print(f"Predictions saved → {preds_path}")
    curve_path = save_pr_curve(y_test, test_scores, model_name)
    print(f"PR curve saved → {curve_path}")

    return results


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniROCKET time-series classification baseline")
    parser.add_argument("--max_len", type=int, default=MAX_LEN_DEFAULT,
                        help=f"Pad/truncate tracks to this many frames (default: {MAX_LEN_DEFAULT})")
    parser.add_argument("--check-deps", action="store_true",
                        help="Verify the aeon transformer import and exit")
    parser.add_argument("--output-dir", default=None,
                        help="Override results directory (default: data/results/six_domain_v1/rocket)")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                        help=(
                            "Worker count for the aeon transformer. Negative values follow "
                            "joblib semantics and are resolved to an explicit numba-safe count "
                            "(default: TRAIT_N_JOBS, ROCKET_N_JOBS, or -1)."
                        ))
    parser.add_argument("--model-name", default=None,
                        help="Model label written to result files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap tracks per split (smoke-testing only)")
    args = parser.parse_args()

    if args.output_dir is not None:
        RESULTS_DIR = Path(args.output_dir)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.check_deps:
        check_dependencies()
    else:
        run(max_len=args.max_len, n_jobs=args.n_jobs,
            model_name=args.model_name, limit=args.limit)
