"""
src/train_transformer.py — Transformer time-series classification baseline (Phase 3)
======================================================================================
A compact Transformer-encoder baseline on raw 2D centroid sequences, added
alongside the FCN and miniROCKET baselines in response to reviewer requests
for a modern Transformer-based time-series classifier (WACV #491 camera-ready
revision). Follows the general encoder framework used for multivariate
time-series representation learning / classification by Zerveas et al.,
"A Transformer-based Framework for Multivariate Time Series Representation
Learning" (KDD 2021): a linear input projection, a learnable positional
embedding, a stack of standard Transformer encoder layers, and a
classification head on the pooled sequence representation. This is a
deliberately small, standard encoder (not a reproduction of Zerveas et al.'s
BatchNorm-based encoder variant) — it represents the architecture family
rather than an exact copy of any one paper's implementation. It is also
motivated by recent strong Transformer-based multivariate TSC methods such as
ConvTran (Foumani et al.), which combines convolutional input encoding with
Transformer attention and specialised positional encodings.

No graph-based baseline is included: each sample here is a single centroid
trajectory with no interacting agents and no intrinsic spatial graph
topology, so a graph model would need to impose an artificial structure
rather than exploit one naturally present in the data.

Uses the SAME held-out test protocol, preprocessing, and evaluation harness
as train_fcn.py and train_rocket.py:
  - Train : folds 0, 1, 2
  - Val   : fold 3  (threshold sweep + early stopping / epoch selection)
  - Test  : fold 4  (held-out, never touched during development)

Gap-filling, smoothing, channel construction (x_norm, y_norm, speed_norm,
dheading), scaling, threshold selection, latency measurement, and bootstrap
CIs are all imported directly from train_fcn.py rather than reimplemented,
so the Transformer, FCN, and miniROCKET consume an identical preprocessed
trajectory and differ only in architecture — this directly addresses the
reviewer concern that the sequence baselines might not receive equivalent
preprocessing to each other or to the engineered-feature pipeline.

Input :  data/tracks/          — per-track CSVs with columns [nframe, x, y]
         data/splits_folds.csv — fold assignments
Output:  data/results/six_domain_v1/transformer/transformer_results.csv
         data/results/six_domain_v1/transformer/transformer_predictions.csv
         data/results/six_domain_v1/transformer/transformer_pr_curve.csv

Training uses linear LR warmup (default: first 10% of optimiser steps) then
cosine decay to 0, rather than train_fcn.py's plain per-epoch cosine decay —
Transformers are known to be warmup-sensitive in a way CNNs generally are
not. Override with --warmup-ratio.

Run from the TRAIT project root (needs PyTorch, same as train_fcn.py):
    python src/train_transformer.py [--max_len 512] [--epochs 60] [--warmup-ratio 0.1] [--check-deps]
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

# Reuse the exact preprocessing / evaluation harness used by the FCN baseline,
# rather than reimplementing gap-filling, smoothing, or channel construction a
# third time. This is the load-bearing choice for preprocessing parity.
from src.train_fcn import (  # noqa: E402
    CHANNELS,
    SEED,
    MAX_LEN_DEFAULT,
    TRACKS_DIR,
    SPLITS_CSV,
    _import_torch,
    configure_torch_runtime,
    resolve_device,
    load_track,
    metric_ci,
    measure_latency,
    model_slug,
)

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
LOADER_SEED = SEED
INITIAL_MODEL_SEED = SEED
FINAL_MODEL_SEED = SEED + 1

# Transformer hyper-parameters — deliberately small, proportionate to the FCN
# baseline's parameter count and appropriate for a ~23k-track training set.
D_MODEL         = 64
N_HEADS         = 4
N_LAYERS        = 3
DIM_FEEDFORWARD = 128
DROPOUT         = 0.1

# Training hyper-parameters — identical to train_fcn.py so the comparison
# isolates architecture, not training budget.
EPOCHS       = 60
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 8

# Linear warmup fraction of total optimiser steps, followed by cosine decay
# to 0 over the remaining steps. Transformers are known to be sensitive to
# missing LR warmup (unlike the FCN's plain CosineAnnealingLR, which doesn't
# need it) — added after the first full run showed a legitimate but weaker-
# than-expected result, to rule out under-optimisation as a confound before
# reporting the Transformer baseline in the paper.
WARMUP_RATIO = 0.1

# Paths (relative to project root)
RESULTS_DIR = Path("data/results/six_domain_v1/transformer")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_pr_curve(y_true, y_score, model_name: str) -> Path:
    """Save precision-recall curve points for plotting.

    Deliberately a local copy of train_fcn.py's helper of the same name
    (rather than an import) because that version closes over train_fcn.py's
    own module-level RESULTS_DIR, which would silently write this script's
    PR curve into the FCN results folder instead of this script's --output-dir.
    """
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


def check_dependencies():
    """Verify the torch import without starting the experiment."""
    torch, _ = _import_torch()
    print(f"PyTorch dependency import OK: {torch.__version__}  ({sys.executable})")


def build_warmup_cosine_schedule(optimizer, torch, warmup_steps: int, total_steps: int):
    """Linear LR warmup for warmup_steps, then cosine decay to 0 over the rest.

    Stepped once per optimiser step (per batch), not once per epoch — unlike
    train_fcn.py's plain per-epoch CosineAnnealingLR, warmup needs step-level
    granularity to ramp smoothly over the first fraction of training rather
    than jumping in whole-epoch increments. warmup_steps and total_steps are
    both expressed in optimiser steps (batches), computed fresh per call to
    train_model() from that call's own epoch budget and loader length, so
    both the initial (early-stopped) and final (fixed-epoch) training phases
    get a schedule correctly scaled to their own step budget.
    """
    import math

    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Transformer model ──────────────────────────────────────────────────────────

def build_transformer(in_channels: int, max_len: int, nn):
    """Compact Transformer encoder for multivariate time-series classification.

    Linear input projection -> learnable positional embedding -> stack of
    standard nn.TransformerEncoderLayer blocks (self-attention + GELU FFN,
    LayerNorm) -> length-masked mean pooling -> linear classification head.
    The padding mask excludes padded frames from both self-attention (via
    src_key_padding_mask) and the final pooling, mirroring the FCN's masked
    global average pooling so padding never contaminates the representation.
    """
    import torch

    class TransformerTSC(nn.Module):
        def __init__(self, c_in: int, d_model: int, nhead: int, num_layers: int,
                     dim_feedforward: int, dropout: float, max_len: int):
            super().__init__()
            self.input_proj = nn.Linear(c_in, d_model)
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x, lengths=None):
            # x: (N, C, T) -> (N, T, C) for batch_first attention
            x = x.transpose(1, 2)
            T = x.size(1)
            h = self.input_proj(x) + self.pos_embedding[:, :T, :]

            if lengths is not None:
                ar = torch.arange(T, device=x.device).unsqueeze(0)          # (1, T)
                pad_mask = ar >= lengths.unsqueeze(1)                        # (N, T) True = padding
            else:
                pad_mask = None

            h = self.encoder(h, src_key_padding_mask=pad_mask)              # (N, T, d_model)

            if lengths is not None:
                valid = (~pad_mask).unsqueeze(-1).to(h.dtype)               # (N, T, 1)
                summed = (h * valid).sum(dim=1)
                denom = valid.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            else:
                pooled = h.mean(dim=1)

            pooled = self.dropout(pooled)
            return self.head(pooled)                                        # (N, 1)

    return TransformerTSC(in_channels, D_MODEL, N_HEADS, N_LAYERS,
                           DIM_FEEDFORWARD, DROPOUT, max_len)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(max_len: int = MAX_LEN_DEFAULT, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
        limit: int | None = None, early_stop_metric: str = "pr_auc",
        model_name: str | None = None, deterministic: bool = False,
        device_name: str = "auto", warmup_ratio: float = WARMUP_RATIO):
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
        model_name = "Transformer_PR_AUC" if early_stop_metric == "pr_auc" else "Transformer"
    print(f"\n{'='*60}")
    print(f" {model_name}  |  max_len={max_len}  |  channels={CHANNELS}  |  device={device}")
    print(f" d_model={D_MODEL}  n_heads={N_HEADS}  n_layers={N_LAYERS}  "
          f"dim_ff={DIM_FEEDFORWARD}  dropout={DROPOUT}")
    print(f" LR schedule: linear warmup ({warmup_ratio:.0%} of steps) -> cosine decay to 0")
    if deterministic:
        print(" Deterministic mode: enabled")
    print(
        f" Seeds: global={SEED}, loader={LOADER_SEED}, "
        f"initial_model={INITIAL_MODEL_SEED}, final_model={FINAL_MODEL_SEED}"
    )
    print(f" Early stopping metric: validation {early_stop_metric.replace('_', '-').upper()}")
    print(f"{'='*60}\n")

    # ── 1. Load splits ──────────────────────────────────────────────────────
    print("Loading splits…")
    splits = pd.read_csv(SPLITS_CSV)
    train_files = list(splits[splits.fold.isin([0, 1, 2])]["filename"])
    val_files   = list(splits[splits.fold == 3]["filename"])
    test_files  = list(splits[splits.fold == 4]["filename"])

    # Ground-truth labels (same resolution logic as train_fcn.py / train_rocket.py)
    try:
        labels_df = pd.read_csv("data/track_classification.csv")
        if "filename" not in labels_df.columns and "filepath" in labels_df.columns:
            labels_df["filename"] = labels_df["filepath"].apply(lambda p: Path(p).name)
    except FileNotFoundError:
        labels_df = splits[["filename"]].copy()
        if "label" in splits.columns:
            labels_df["label"] = splits["label"]
        else:
            raise FileNotFoundError(
                "Cannot find data/track_classification.csv — "
                "ensure it has columns [filename/filepath, label] where label ∈ {insect, not_insect}"
            )

    label_map = dict(zip(labels_df["filename"], labels_df["label"]))
    binary = {f: 1 if str(v).lower() in ("insect", "1") else 0
              for f, v in label_map.items()}

    # ── 2. Load and featurise tracks (identical to train_fcn.py) ────────────
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
        # Chunk size deliberately small (not FCN's 1024): self-attention memory
        # scales with batch_size * n_heads * max_len^2, so a large eval chunk
        # at max_len=512 can spike memory far above the training batch size
        # even under no_grad(). Reuses the training batch_size as a safe cap.
        model.eval()
        out = []
        eval_chunk = max(1, min(batch_size, 64))
        for i in range(0, len(X), eval_chunk):
            xb = torch.from_numpy(X[i:i+eval_chunk]).to(device)
            lb = torch.from_numpy(L[i:i+eval_chunk]).to(device)
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
        model = build_transformer(C, max_len, nn).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        steps_per_epoch = max(1, len(loader_))
        total_steps = max(1, n_epochs * steps_per_epoch)
        warmup_steps = max(1, int(round(warmup_ratio * total_steps)))
        sched = build_warmup_cosine_schedule(opt, torch, warmup_steps, total_steps)

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
                sched.step()  # per-step (not per-epoch): warmup needs step-level granularity
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
    print("Training Transformer on train folds (0-2) …")
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

    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'─'*40}")
    print(f"  Model      : {model_name}")
    print(f"  max_len    : {max_len}")
    print(f"  Params     : {n_params:,}")
    print(f"  Stop metric: {early_stop_metric}")
    print(f"  Threshold  : {best_thr:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Test F1    : {f1:.4f}  95% CI [{ci_lo:.3f}–{ci_hi:.3f}]")
    print(f"  PR-AUC     : {prauc:.4f}")
    print(f"  Latency    : {latency['latency_mean_ms']:.3f} ms/sample")
    print(f"{'─'*40}\n")

    # ── 7. Save outputs (schema identical to train_fcn.py / train_rocket.py) ─
    results = pd.DataFrame([{
        "model":     model_name,
        "max_len":   max_len,
        "channels":  ",".join(CHANNELS),
        "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD, "dropout": DROPOUT,
        "warmup_ratio": warmup_ratio,
        "batch_size": int(batch_size),   # recorded so each run is self-documenting
        "n_params": int(n_params),
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
    results_path = RESULTS_DIR / "transformer_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")

    preds = pd.DataFrame({
        "filename": fnames_test,
        "y_true":   y_test.astype(int),
        "y_pred":   y_pred,
        "score":    test_scores,
    })
    preds_path = RESULTS_DIR / "transformer_predictions.csv"
    preds.to_csv(preds_path, index=False)
    print(f"Predictions saved → {preds_path}")
    curve_path = save_pr_curve(y_test, test_scores, model_name)
    print(f"PR curve saved → {curve_path}")

    return results


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer time-series classification baseline")
    parser.add_argument("--max_len", type=int, default=MAX_LEN_DEFAULT,
                        help=f"Pad/truncate tracks to this many frames (default: {MAX_LEN_DEFAULT}; "
                             "padding is masked out of attention and pooling)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap tracks per split (smoke-testing only)")
    parser.add_argument("--early-stop-metric", choices=["f1", "pr_auc"], default="pr_auc",
                        help="Validation metric used for early stopping (default: pr_auc)")
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO,
                        help=f"Fraction of optimiser steps spent on linear LR warmup before "
                             f"cosine decay (default: {WARMUP_RATIO})")
    parser.add_argument("--model-name", default=None,
                        help="Model label written to result files (default: Transformer or Transformer_PR_AUC)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic PyTorch settings for final reproducibility runs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default=os.environ.get("TRAIT_TRANSFORMER_DEVICE", "auto"),
                        help="Training device (default: TRAIT_TRANSFORMER_DEVICE or auto)")
    parser.add_argument("--check-deps", action="store_true",
                        help="Verify the torch import and exit")
    parser.add_argument("--output-dir", default=None,
                        help="Override results directory (default: data/results/six_domain_v1/transformer)")
    args = parser.parse_args()

    if args.output_dir is not None:
        RESULTS_DIR = Path(args.output_dir)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.check_deps:
        check_dependencies()
    else:
        run(max_len=args.max_len, epochs=args.epochs,
            batch_size=args.batch_size, limit=args.limit,
            early_stop_metric=args.early_stop_metric,
            model_name=args.model_name,
            deterministic=args.deterministic or os.environ.get("TRAIT_DETERMINISTIC") == "1",
            device_name=args.device, warmup_ratio=args.warmup_ratio)
