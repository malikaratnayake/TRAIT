"""
src/train_fcn.py — Fully Convolutional Network (1D-CNN) baseline (Phase 3)
=========================================================================
End-to-end deep learning baseline on raw 2D centroid sequences, complementing
miniROCKET as a second learned reference point. Implements the Fully
Convolutional Network (FCN) of Wang, Yan & Oates, "Time Series Classification
from Scratch with Deep Neural Networks" (IJCNN 2017) — the standard deep TSC
baseline: three Conv1d blocks (128 → 256 → 128) with BatchNorm + ReLU, global
average pooling, and a single linear output. No appearance information is used;
the network learns directly from the trajectory, so it isolates the value of the
hand-crafted six-domain representation against a purely data-driven model.

Uses the SAME held-out test protocol as every other model:
  - Train : folds 0, 1, 2
  - Val   : fold 3  (threshold sweep + early stopping / epoch selection)
  - Test  : fold 4  (held-out, never touched during development)

Input :  data/tracks/          — per-track CSVs with columns [nframe, x, y]
         data/splits_folds.csv — fold assignments
Output:  data/results/six_domain_v1/fcn/fcn_results.csv
         data/results/six_domain_v1/fcn/fcn_predictions.csv
         data/results/six_domain_v1/fcn/fcn_pr_curve.csv

Run from the TRAIT project root (needs PyTorch in $PYTHON):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python src/train_fcn.py [--max_len 512] [--epochs 60] [--early-stop-metric pr_auc] [--check-deps]

The four input channels (x_norm, y_norm, speed_norm, dheading) match
train_rocket.py exactly, so the FCN and miniROCKET consume an identical
trajectory representation without exposing absolute image-frame position.

Preprocessing parity (2026-08 fix): gap-filling now reuses the exact
linear-interpolation-then-SG-smoothing logic used by src/generate_features.py
to build the engineered-feature set (via src/utils/smoothing.smooth_coords),
instead of dropping missing-detection frames before smoothing. Previously
this script discarded NaN rows prior to smoothing, which compressed the
effective time base at every gap and fed the derived speed/heading channels
spurious jumps at gap boundaries. Both baselines and the engineered features
now consume an identical gap-filled trajectory; the only remaining difference
is whether behaviour-informed features are computed from it or raw channels
are fed directly to the model.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.smoothing import smooth_coords  # same fn used by generate_features.py

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
SG_WINDOW = 11
SG_POLY   = 2
CHANNELS  = ["x_norm", "y_norm", "speed_norm", "dheading"]  # identical to train_rocket.py
SEED      = 42
LOADER_SEED = SEED
INITIAL_MODEL_SEED = SEED
FINAL_MODEL_SEED = SEED + 1
# Default sequence length. Median track is ~150 frames (p95 ≈ 503), so 128
# truncated the majority of tracks; 512 covers ~95% of tracks in full. Padding
# is masked out of the global pooling (see FCN.forward), so the larger window
# recovers long-track signal without diluting short tracks.
MAX_LEN_DEFAULT = 512

# Training hyper-parameters (FCN, Wang et al. 2017 settings, lightly adapted)
EPOCHS      = 60
BATCH_SIZE  = 128
LR          = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE    = 8          # early-stopping patience on the selected validation metric

# Default paths (relative to project root); overridden by the CLI options.
TRACKS_DIR  = Path("data/tracks")
SPLITS_CSV  = Path("data/splits_folds.csv")
LABELS_CSV  = Path("data/track_classification.csv")
RESULTS_DIR = Path("data/results/trait_results/fcn_prauc")


# ─── Lazy torch import ─────────────────────────────────────────────────────────

def _import_torch():
    """Import torch on demand so the file loads even when torch is absent."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        return torch, nn
    except ImportError as exc:
        print(f"ERROR: PyTorch is required for the FCN baseline but is not installed: {exc}")
        print("Install it into the pipeline interpreter, e.g.:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        raise


def check_dependencies():
    """Verify the torch import without starting the experiment."""
    torch, _ = _import_torch()
    print(f"PyTorch dependency import OK: {torch.__version__}  ({sys.executable})")


def configure_torch_runtime(torch, deterministic: bool) -> None:
    """Apply final-run reproducibility settings where PyTorch supports them."""
    if not deterministic:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(torch, requested: str):
    """Resolve a requested device name without silently using unavailable hardware."""
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
    return torch.device(requested)


# ─── Helpers (identical representation to train_rocket.py) ─────────────────────

def interpolate_and_smooth(x_raw: np.ndarray, y_raw: np.ndarray):
    """Gap-fill via linear interpolation, then SG-smooth.

    Reuses smooth_coords (src/utils/smoothing.py) with the exact same
    interpolation step used by src/generate_features.py's on-the-fly
    smoothing path (main(), ~line 2213): NaN gaps are linearly interpolated
    over row index before the SG filter is applied, so the sequence baselines
    consume the identical gap-filled trajectory as the engineered features,
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


def load_track(csv_path: Path, max_len: int):
    """
    Load a track CSV, apply SG smoothing, compute the four derived channels,
    and return ``(channels, real_len)`` where ``channels`` is a (C, max_len)
    array padded / truncated to max_len and ``real_len`` is the number of
    genuine (non-padded) frames it contains (= min(T, max_len)).

    Returns None if the file is missing or has fewer than 3 valid (non-NaN)
    detections. Gaps between valid detections are linearly interpolated
    (matching the engineered-feature pipeline) rather than dropped, so the
    row count / effective time base is preserved across the track's full
    recorded span.
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

    dx = np.diff(sx, prepend=sx[0])
    dy = np.diff(sy, prepend=sy[0])
    speed = np.sqrt(dx**2 + dy**2)

    heading = np.arctan2(dy, dx)
    dheading = np.diff(heading, prepend=heading[0])
    dheading = (dheading + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]

    sx_norm, sy_norm, speed_norm = normalise_track_frame(sx, sy, speed)

    channels = np.stack([sx_norm, sy_norm, speed_norm, dheading], axis=0)  # (4, T)

    T = channels.shape[1]
    real_len = min(T, max_len)
    if T >= max_len:
        channels = channels[:, :max_len]
    else:
        pad = np.zeros((channels.shape[0], max_len - T), dtype=np.float32)
        channels = np.concatenate([channels, pad], axis=1)

    return channels.astype(np.float32), int(real_len)


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


def measure_latency(predict_fn, X_sample: np.ndarray, n_reps: int = 500, warmup: int = 50) -> dict:
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


def model_slug(model_name: str) -> str:
    """Return a stable lowercase filename prefix for model-specific artifacts."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name).strip("_")


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
    path = RESULTS_DIR / f"{model_slug(model_name)}_pr_curve.csv"
    curve.to_csv(path, index=False)
    return path


# ─── FCN model (Wang et al. 2017) ──────────────────────────────────────────────

def build_fcn(in_channels: int, nn):
    """Fully Convolutional Network with length-masked global average pooling.

    Architecture matches Wang et al. (2017): three Conv1d blocks
    (128 → 256 → 128) with BatchNorm + ReLU, then global average pooling and a
    single linear output. Unlike a plain ``AdaptiveAvgPool1d``, the pooling here
    averages each sample only over its genuine (non-padded) frames, so
    zero-padding added to reach a common length does not dilute the pooled
    representation of short tracks.
    """
    import torch

    class FCN(nn.Module):
        def __init__(self, c_in: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(c_in, 128, kernel_size=8, padding="same"),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=5, padding="same"),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding="same"),
                nn.BatchNorm1d(128), nn.ReLU(),
            )
            self.head = nn.Linear(128, 1)

        def forward(self, x, lengths=None):
            h = self.features(x)                      # (N, 128, T)
            if lengths is None:
                pooled = h.mean(dim=2)
            else:
                T = h.size(2)
                ar = torch.arange(T, device=h.device).unsqueeze(0)     # (1, T)
                mask = (ar < lengths.unsqueeze(1)).to(h.dtype)         # (N, T)
                mask = mask.unsqueeze(1)                               # (N, 1, T)
                summed = (h * mask).sum(dim=2)                         # (N, 128)
                denom = mask.sum(dim=2).clamp(min=1.0)                 # (N, 1)
                pooled = summed / denom
            return self.head(pooled)                  # (N, 1)

    return FCN(in_channels)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(max_len: int = MAX_LEN_DEFAULT, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
        limit: int | None = None, early_stop_metric: str = "f1",
        model_name: str | None = None, deterministic: bool = False,
        device_name: str = "auto"):
    if early_stop_metric not in {"f1", "pr_auc"}:
        raise ValueError("early_stop_metric must be either 'f1' or 'pr_auc'")

    torch, nn = _import_torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    configure_torch_runtime(torch, deterministic)
    device = resolve_device(torch, device_name)
    torch_num_threads = torch.get_num_threads()
    try:
        torch_num_interop_threads = torch.get_num_interop_threads()
    except RuntimeError:
        torch_num_interop_threads = None

    if model_name is None:
        model_name = "FCN_PR_AUC" if early_stop_metric == "pr_auc" else "FCN"
    print(f"\n{'='*60}")
    print(f" {model_name}  |  max_len={max_len}  |  channels={CHANNELS}  |  device={device}")
    if deterministic:
        print(" Deterministic mode: enabled")
    print(
        f" Seeds: global={SEED}, loader={LOADER_SEED}, "
        f"initial_model={INITIAL_MODEL_SEED}, final_model={FINAL_MODEL_SEED}"
    )
    print(f" Early stopping metric: validation {early_stop_metric.replace('_', '-').upper()}")
    print(f"{'='*60}\n")

    # ── 1. Load splits ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading splits…")
    splits = pd.read_csv(SPLITS_CSV)
    train_files = list(splits[splits.fold.isin([0, 1, 2])]["filename"])
    val_files   = list(splits[splits.fold == 3]["filename"])
    test_files  = list(splits[splits.fold == 4]["filename"])

    # Ground-truth labels (same resolution logic as train_rocket.py)
    try:
        labels_df = pd.read_csv(LABELS_CSV)
        if "filename" not in labels_df.columns and "filepath" in labels_df.columns:
            labels_df["filename"] = labels_df["filepath"].apply(lambda p: Path(p).name)
    except FileNotFoundError:
        labels_df = splits[["filename"]].copy()
        if "label" in splits.columns:
            labels_df["label"] = splits["label"]
        else:
            raise FileNotFoundError(
                f"Cannot find {LABELS_CSV} — "
                "ensure it has columns [filename/filepath, label] where label ∈ {insect, not_insect}"
            )

    label_map = dict(zip(labels_df["filename"], labels_df["label"]))
    binary = {f: 1 if str(v).lower() in ("insect", "1") else 0
              for f, v in label_map.items()}

    # ── 2. Load and featurise tracks ────────────────────────────────────────
    def build_array(file_list, desc: str):
        if limit is not None:
            file_list = file_list[:limit]
        X, y, L, fnames = [], [], [], []
        missing = 0
        for fn in sorted(file_list):
            loaded = load_track(TRACKS_DIR / fn, max_len)
            lbl = binary.get(fn)
            if loaded is None or lbl is None:
                missing += 1
                continue
            arr, real_len = loaded
            X.append(arr)
            y.append(lbl)
            L.append(real_len)
            fnames.append(fn)
        X = np.stack(X, axis=0)            # (N, C, T)
        y = np.array(y, dtype=np.float32)
        L = np.array(L, dtype=np.int64)    # genuine (non-padded) length per track
        print(f"  {desc}: {len(X)} tracks  (skipped {missing})   insect={y.mean():.1%}")
        return X, y, L, fnames

    print("Loading tracks (this may take ~1-2 min for 28k tracks)…")
    t0 = time.time()
    X_train, y_train, L_train, _ = build_array(train_files, "train")
    X_val,   y_val,   L_val,   _ = build_array(val_files,   "val  ")
    X_test,  y_test,  L_test,  fnames_test = build_array(test_files, "test ")
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # ── 3. Per-channel standardisation (fit on train REAL frames only) ──────
    # StandardScaler over (sample·time) for each channel — essential for a CNN.
    # Padded frames are excluded from the fit so the statistics reflect genuine
    # trajectory values, not the zero padding.
    C = X_train.shape[1]
    T_len = X_train.shape[2]
    scaler = StandardScaler()
    _flat_train = X_train.transpose(0, 2, 1).reshape(-1, C)
    _real_train = (np.arange(T_len)[None, :] < L_train[:, None]).reshape(-1)
    scaler.fit(_flat_train[_real_train])

    def apply_scaler(X):
        N, c, T = X.shape
        flat = scaler.transform(X.transpose(0, 2, 1).reshape(-1, c))
        return flat.reshape(N, T, c).transpose(0, 2, 1).astype(np.float32)

    X_train = apply_scaler(X_train)
    X_val   = apply_scaler(X_val)
    X_test  = apply_scaler(X_test)

    # ── 4. Torch tensors / loaders ──────────────────────────────────────────
    from torch.utils.data import DataLoader, TensorDataset

    def loader(X, y, L, shuffle, loader_seed: int = LOADER_SEED):
        ds = TensorDataset(torch.from_numpy(X),
                           torch.from_numpy(y),
                           torch.from_numpy(L))
        g = torch.Generator().manual_seed(loader_seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)

    train_loader = loader(X_train, y_train, L_train, True)

    def make_criterion(y_values):
        n_pos = int(np.sum(y_values))
        n_neg = int(len(y_values) - n_pos)
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)],
            dtype=torch.float32,
            device=device,
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_criterion = make_criterion(y_train)

    @torch.no_grad()
    def predict_proba(model, X, L):
        model.eval()
        out = []
        for i in range(0, len(X), 1024):
            xb = torch.from_numpy(X[i:i+1024]).to(device)
            lb = torch.from_numpy(L[i:i+1024]).to(device)
            out.append(torch.sigmoid(model(xb, lb).squeeze(-1)).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def train_model(
        loader_,
        n_epochs,
        criterion_,
        *,
        model_seed: int,
        val_X=None,
        val_y=None,
        val_L=None,
    ):
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_fcn(C, nn).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
        best_metric, best_state, patience = -np.inf, None, 0
        best_epoch = n_epochs
        metric_label = "val PR-AUC" if early_stop_metric == "pr_auc" else "val F1"
        for ep in range(1, n_epochs + 1):
            model.train()
            for xb, yb, lb in loader_:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                opt.zero_grad()
                loss = criterion_(model(xb, lb).squeeze(-1), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            if val_X is not None:
                vp = predict_proba(model, val_X, val_L)
                if early_stop_metric == "pr_auc":
                    metric_value = average_precision_score(val_y, vp)
                else:
                    metric_value = max(
                        f1_score(val_y, (vp >= t).astype(int), zero_division=0)
                        for t in np.linspace(0.2, 0.8, 31)
                    )
                if metric_value > best_metric:
                    best_metric, patience = metric_value, 0
                    best_epoch = ep
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        print(f"  Early stop at epoch {ep}  (best {metric_label}={best_metric:.4f})")
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, best_metric

    # ── 5. Train on folds 0-2, tune threshold on fold 3 ─────────────────────
    print("Training FCN on train folds (0-2) …")
    t0 = time.time()
    model, selected_epochs, selected_metric = train_model(
        train_loader,
        epochs,
        train_criterion,
        model_seed=INITIAL_MODEL_SEED,
        val_X=X_val,
        val_y=y_val,
        val_L=L_val,
    )
    print(f"  Trained in {time.time()-t0:.1f}s")
    print(f"  Selected epoch count: {selected_epochs}  ({early_stop_metric}={selected_metric:.4f})")

    val_scores = predict_proba(model, X_val, L_val)
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 181):
        f = f1_score(y_val, (val_scores >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    print(f"  Val-optimal threshold: {best_thr:.4f}  (val F1 = {best_f1:.4f})")

    # ── 6. Retrain on train+val, evaluate on held-out test fold 4 ───────────
    print(f"\nRetraining on train+val for final evaluation ({selected_epochs} epochs)…")
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    L_trainval = np.concatenate([L_train, L_val], axis=0)
    final_loader = loader(X_trainval, y_trainval, L_trainval, True)
    final_criterion = make_criterion(y_trainval)
    t0 = time.time()
    model, _, _ = train_model(
        final_loader,
        selected_epochs,
        final_criterion,
        model_seed=FINAL_MODEL_SEED,
    )
    print(f"  Retrained in {time.time()-t0:.1f}s")

    test_scores = predict_proba(model, X_test, L_test)
    y_pred = (test_scores >= best_thr).astype(int)

    sample_len = int(L_test[0])   # genuine length of the latency probe track

    def predict_one_track(x):
        xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        lb = torch.full((xb.shape[0],), sample_len, dtype=torch.long, device=device)
        with torch.no_grad():
            model.eval()
            return (torch.sigmoid(model(xb, lb).squeeze(-1)).cpu().numpy() >= best_thr).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    prauc     = average_precision_score(y_test, test_scores)
    ci = metric_ci(y_test, y_pred, test_scores)
    ci_lo, ci_hi = ci["f1_lo"], ci["f1_hi"]
    latency   = measure_latency(predict_one_track, X_test[0])

    print(f"\n{'─'*40}")
    print(f"  Model      : {model_name}")
    print(f"  max_len    : {max_len}")
    print(f"  Stop metric: {early_stop_metric}")
    print(f"  Threshold  : {best_thr:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Test F1    : {f1:.4f}  95% CI [{ci_lo:.3f}–{ci_hi:.3f}]")
    print(f"  PR-AUC     : {prauc:.4f}")
    print(f"  Latency    : {latency['latency_mean_ms']:.3f} ms/sample")
    print(f"{'─'*40}\n")

    # ── 7. Save outputs (schema identical to train_rocket.py) ───────────────
    results = pd.DataFrame([{
        "model":     model_name,
        "max_len":   max_len,
        "channels":  ",".join(CHANNELS),
        "early_stop_metric": early_stop_metric,
        "seed": SEED,
        "loader_seed": LOADER_SEED,
        "initial_model_seed": INITIAL_MODEL_SEED,
        "final_model_seed": FINAL_MODEL_SEED,
        "deterministic": bool(deterministic),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_num_threads": torch_num_threads,
        "torch_num_interop_threads": torch_num_interop_threads,
        "selected_epochs": selected_epochs,
        "validation_selection_score": round(selected_metric, 4),
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
    results_path = RESULTS_DIR / "fcn_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")

    preds = pd.DataFrame({
        "filename": fnames_test,
        "y_true":   y_test.astype(int),
        "y_pred":   y_pred,
        "score":    test_scores,
    })
    preds_path = RESULTS_DIR / "fcn_predictions.csv"
    preds.to_csv(preds_path, index=False)
    print(f"Predictions saved → {preds_path}")
    curve_path = save_pr_curve(y_test, test_scores, model_name)
    print(f"PR curve saved → {curve_path}")

    return results


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCN (1D-CNN) time-series classification baseline")
    parser.add_argument("--max_len", type=int, default=MAX_LEN_DEFAULT,
                        help=f"Pad/truncate tracks to this many frames (default: {MAX_LEN_DEFAULT}; "
                             "padding is masked out of global pooling)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap tracks per split (smoke-testing only)")
    parser.add_argument("--early-stop-metric", choices=["f1", "pr_auc"], default="pr_auc",
                        help="Validation metric used for early stopping (default: pr_auc)")
    parser.add_argument("--model-name", default=None,
                        help="Model label written to result files (default: FCN or FCN_PR_AUC)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic PyTorch settings for final reproducibility runs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default=os.environ.get("TRAIT_FCN_DEVICE", "auto"),
                        help="Training device (default: TRAIT_FCN_DEVICE or auto)")
    parser.add_argument("--check-deps", action="store_true",
                        help="Verify the torch import and exit")
    parser.add_argument("--output-dir", default=None,
                        help=f"Results directory (default: {RESULTS_DIR})")
    parser.add_argument("--tracks-dir", default=str(TRACKS_DIR),
                        help=f"Directory of per-track CSVs (default: {TRACKS_DIR})")
    parser.add_argument("--splits-csv", default=str(SPLITS_CSV),
                        help=f"Fold assignments from create_splits.py (default: {SPLITS_CSV})")
    parser.add_argument("--labels-csv", default=str(LABELS_CSV),
                        help=f"Track labels with filepath and label columns (default: {LABELS_CSV})")
    args = parser.parse_args()

    TRACKS_DIR = Path(args.tracks_dir)
    SPLITS_CSV = Path(args.splits_csv)
    LABELS_CSV = Path(args.labels_csv)
    if args.output_dir is not None:
        RESULTS_DIR = Path(args.output_dir)

    if args.check_deps:
        check_dependencies()
    else:
        run(max_len=args.max_len, epochs=args.epochs,
            batch_size=args.batch_size, limit=args.limit,
            early_stop_metric=args.early_stop_metric,
            model_name=args.model_name,
            deterministic=args.deterministic or os.environ.get("TRAIT_DETERMINISTIC") == "1",
            device_name=args.device)
