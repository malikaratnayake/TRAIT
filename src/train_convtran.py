"""
src/train_convtran.py — ConvTran time-series classification baseline (Phase 3)
======================================================================================
Implementation of ConvTran (Foumani et al., "Improving position encoding of
transformers for multivariate time series classification", DMKD 2024),
written from the architecture description and the official reference code
(github.com/Navidfoumani/ConvTran) so it can share this project's data
pipeline. Three components define ConvTran and are reproduced here:

  1. A two-stage Conv2d "disjoint CNN" input embedding: a temporal
     convolution applied independently per input channel (kernel [1, 8]),
     followed by a channel-collapsing convolution (kernel [C, 1]) that mixes
     across channels once. This gives the tokeniser a local, convolutional
     inductive bias that a linear input projection lacks.
  2. tAPE (time-scaled Absolute Positional Encoding): a sinusoidal encoding
     whose frequency is scaled by d_model/max_len so its resolution matches
     the sequence length.
  3. eRPE (efficient Relative Position Encoding): a learnable scalar bias
     per (query, key) relative distance, added to attention logits before
     softmax.

  Adaptation for variable-length tracks: the original ConvTran targets
  UCR/UEA benchmarks where series have a uniform length, so its attention and
  pooling have no notion of padding. Our tracks are zero-padded to max_len,
  so this version adds an explicit key-padding mask to the eRPE attention
  (padded keys excluded via -inf before softmax) and to the final pooling
  (masked mean instead of AdaptiveAvgPool1d), as train_fcn.py does for the
  1D-CNN. The original is single-block by default (one conv tokeniser, one
  eRPE attention block, one feed-forward block); this is kept as the default
  (--n-layers 1).

Uses the same held-out test protocol, preprocessing and evaluation harness
as train_fcn.py and train_rocket.py:
  - Train : folds 0, 1, 2
  - Val   : fold 3  (threshold sweep + early stopping / epoch selection)
  - Test  : fold 4  (held-out)

Gap-filling, smoothing, channel construction (x_norm, y_norm, speed_norm,
dheading), threshold selection, latency measurement and bootstrap CIs are
imported from train_fcn.py, so ConvTran consumes the same preprocessed
trajectory as the 1D-CNN and miniROCKET and differs only in architecture.

Architecture hyperparameters (emb_size=16, num_heads=8, dim_feedforward=256,
dropout=0.01) are ConvTran's published defaults. The training budget (AdamW,
lr 1e-3, weight decay 1e-4, batch size 128, at most 60 epochs, early-stopping
patience 8) matches the 1D-CNN. The original ConvTran uses RAdam with batch
size 16; batch size 128 is used here for parity with the 1D-CNN, and
src/convtran_batch_size_study.py checks that the result is stable across
batch sizes. --warmup-ratio defaults to 0 because the original reports no
learning-rate warmup.

Input :  data/tracks/          — per-track CSVs with columns [nframe, x, y]
         data/splits_folds.csv — fold assignments
Output:  <output-dir>/convtran_results.csv
         <output-dir>/convtran_predictions.csv
         <output-dir>/<model-name>_pr_curve.csv

Run from the TRAIT project root (needs PyTorch):
    python src/train_convtran.py [--max_len 512] [--epochs 60] [--check-deps]
"""

import argparse
import math
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

# Reuse the exact preprocessing / evaluation harness used by the other
# sequence baselines, rather than reimplementing gap-filling, smoothing, or
# channel construction a fourth time. This is the load-bearing choice for
# preprocessing parity across all sequence baselines.
from src.train_fcn import (  # noqa: E402
    CHANNELS,
    SEED,
    MAX_LEN_DEFAULT,
    TRACKS_DIR,
    SPLITS_CSV,
    LABELS_CSV,
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

# ConvTran architecture hyper-parameters — ConvTran's own published defaults
# (Foumani et al., official repo main.py).
EMB_SIZE        = 16
NUM_HEADS       = 8
N_LAYERS        = 1     # published ConvTran is single-block; override to stack
DIM_FEEDFORWARD = 256
DROPOUT         = 0.01
CONV_KERNEL     = 8     # temporal conv kernel width in the disjoint-CNN embedding

# Training hyper-parameters — identical to train_fcn.py's 1D-CNN.
EPOCHS       = 60
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 8

# No warmup by default: the original ConvTran uses RAdam (warmup-free by
# design) and reports no LR warmup. Exposed for experimentation only.
WARMUP_RATIO = 0.0

# Default results directory (relative to project root); overridden by --output-dir.
RESULTS_DIR = Path("data/results/trait_results/convtran_prauc")


def save_pr_curve(y_true, y_score, model_name: str) -> Path:
    """Save precision-recall curve points for plotting.

    Local copy (not an import): train_fcn.py's helper writes to train_fcn's
    own RESULTS_DIR and would misplace this file.
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


# ─── ConvTran model ─────────────────────────────────────────────────────────

def build_convtran(in_channels: int, max_len: int, nn, *,
                    emb_size: int = EMB_SIZE, num_heads: int = NUM_HEADS,
                    n_layers: int = N_LAYERS, dim_feedforward: int = DIM_FEEDFORWARD,
                    dropout: float = DROPOUT, conv_kernel: int = CONV_KERNEL):
    """ConvTran: disjoint-CNN tokeniser + tAPE + eRPE attention.

    Adapted with an explicit padding mask (see module docstring) so padded
    frames are excluded from both attention and the final pooled
    representation, unlike the reference implementation's fixed-length
    UCR/UEA setting.
    """
    import torch

    class tAPE(nn.Module):
        """Time-scaled Absolute Positional Encoding (Foumani et al., 2024)."""

        def __init__(self, d_model: int, max_len: int):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
            pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            T = x.size(1)
            return x + self.pe[:, :T, :]

    class RelativeScalarAttention(nn.Module):
        """eRPE: multi-head self-attention with a learnable scalar relative-
        position bias per (query, key) distance, plus an explicit
        key-padding mask (see module docstring for why the mask is an
        addition over the reference implementation)."""

        def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float):
            super().__init__()
            assert d_model % n_heads == 0, "emb_size must be divisible by num_heads"
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.scale = self.d_head ** -0.5
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.to_out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

            # Learnable scalar bias per relative distance in [-(max_len-1), max_len-1],
            # indexed by (query_pos - key_pos) + (max_len - 1).
            self.relative_bias_table = nn.Parameter(torch.zeros(2 * max_len - 1, n_heads))
            coords = torch.arange(max_len)
            relative_index = coords[:, None] - coords[None, :] + (max_len - 1)  # (T, T)
            self.register_buffer("relative_index", relative_index)

        def forward(self, x, key_padding_mask=None):
            B, T, _ = x.shape
            q = self.q(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
            k = self.k(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
            v = self.v(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

            rel_idx = self.relative_index[:T, :T]
            bias = self.relative_bias_table[rel_idx]        # (T, T, H)
            bias = bias.permute(2, 0, 1).unsqueeze(0)        # (1, H, T, T)
            attn = attn + bias

            if key_padding_mask is not None:
                # key_padding_mask: (B, T) True = padding
                mask = key_padding_mask[:, None, None, :]     # (B, 1, 1, T)
                attn = attn.masked_fill(mask, float("-inf"))

            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)                       # (B, H, T, d_head)
            out = out.transpose(1, 2).reshape(B, T, -1)
            return self.to_out(out)

    class ConvTranBlock(nn.Module):
        """One eRPE-attention + feed-forward block, pre-tokenised input in."""

        def __init__(self, d_model: int, n_heads: int, max_len: int,
                     dim_feedforward: int, dropout: float):
            super().__init__()
            self.attn = RelativeScalarAttention(d_model, n_heads, max_len, dropout)
            self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

        def forward(self, x, pad_mask=None):
            att = x + self.attn(x, key_padding_mask=pad_mask)
            att = self.norm1(att)
            out = att + self.ff(att)
            out = self.norm2(out)
            return out

    class ConvTranTSC(nn.Module):
        def __init__(self, c_in: int, emb_size: int, n_heads: int, n_layers: int,
                     dim_feedforward: int, dropout: float, max_len: int, conv_kernel: int):
            super().__init__()
            # Disjoint-CNN tokeniser: temporal conv per channel, then a
            # single channel-mixing conv — the local inductive bias that a
            # plain linear projection lacks.
            self.embed1 = nn.Sequential(
                nn.Conv2d(1, emb_size * 4, kernel_size=(1, conv_kernel), padding="same"),
                nn.BatchNorm2d(emb_size * 4),
                nn.GELU(),
            )
            self.embed2 = nn.Sequential(
                nn.Conv2d(emb_size * 4, emb_size, kernel_size=(c_in, 1), padding=0),
                nn.BatchNorm2d(emb_size),
                nn.GELU(),
            )
            self.pos = tAPE(emb_size, max_len)
            self.blocks = nn.ModuleList([
                ConvTranBlock(emb_size, n_heads, max_len, dim_feedforward, dropout)
                for _ in range(n_layers)
            ])
            self.head = nn.Linear(emb_size, 1)

        def forward(self, x, lengths=None):
            # x: (N, C, T)
            B, C, T = x.shape
            h = x.unsqueeze(1)                 # (N, 1, C, T)
            h = self.embed1(h)                  # (N, 4*emb, C, T)
            h = self.embed2(h).squeeze(2)        # (N, emb, T)
            h = h.permute(0, 2, 1)               # (N, T, emb)

            if lengths is not None:
                ar = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
                pad_mask = ar >= lengths.unsqueeze(1)                 # (N, T) True = padding
            else:
                pad_mask = None

            h = self.pos(h)
            for block in self.blocks:
                h = block(h, pad_mask=pad_mask)

            if lengths is not None:
                valid = (~pad_mask).unsqueeze(-1).to(h.dtype)   # (N, T, 1)
                summed = (h * valid).sum(dim=1)
                denom = valid.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            else:
                pooled = h.mean(dim=1)

            return self.head(pooled)             # (N, 1)

    return ConvTranTSC(in_channels, emb_size, num_heads, n_layers,
                        dim_feedforward, dropout, max_len, conv_kernel)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(max_len: int = MAX_LEN_DEFAULT, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
        limit: int | None = None, early_stop_metric: str = "pr_auc",
        model_name: str | None = None, deterministic: bool = False,
        device_name: str = "auto", warmup_ratio: float = WARMUP_RATIO,
        emb_size: int = EMB_SIZE, num_heads: int = NUM_HEADS, n_layers: int = N_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD, dropout: float = DROPOUT):
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
        model_name = "ConvTran_PR_AUC" if early_stop_metric == "pr_auc" else "ConvTran"
    print(f"\n{'='*60}")
    print(f" {model_name}  |  max_len={max_len}  |  channels={CHANNELS}  |  device={device}")
    print(f" emb_size={emb_size}  n_heads={num_heads}  n_layers={n_layers}  "
          f"dim_ff={dim_feedforward}  dropout={dropout}")
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading splits…")
    splits = pd.read_csv(SPLITS_CSV)
    train_files = list(splits[splits.fold.isin([0, 1, 2])]["filename"])
    val_files   = list(splits[splits.fold == 3]["filename"])
    test_files  = list(splits[splits.fold == 4]["filename"])

    # Ground-truth labels (same resolution logic as the other baselines)
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
        # Small eval chunk (not FCN's 1024): self-attention memory scales
        # with batch_size * n_heads * max_len^2, so a large eval chunk at
        # max_len=512 can spike memory even under no_grad().
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
        model = build_convtran(
            C, max_len, nn,
            emb_size=emb_size, num_heads=num_heads, n_layers=n_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        ).to(device)
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
                sched.step()  # stepped per batch, not per epoch
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
    print("Training ConvTran on train folds (0-2) …")
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

    # ── 7. Save outputs (schema aligned with the other sequence baselines) ──
    results = pd.DataFrame([{
        "model":     model_name,
        "max_len":   max_len,
        "channels":  ",".join(CHANNELS),
        "emb_size": emb_size, "n_heads": num_heads, "n_layers": n_layers,
        "dim_feedforward": dim_feedforward, "dropout": dropout,
        "warmup_ratio": warmup_ratio,
        # Recorded so each run is self-documenting: batch size is a deliberate
        # deviation from ConvTran's published default (16) and is the subject
        # of src/convtran_batch_size_study.py.
        "batch_size": int(batch_size),
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
    results_path = RESULTS_DIR / "convtran_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")

    preds = pd.DataFrame({
        "filename": fnames_test,
        "y_true":   y_test.astype(int),
        "y_pred":   y_pred,
        "score":    test_scores,
    })
    preds_path = RESULTS_DIR / "convtran_predictions.csv"
    preds.to_csv(preds_path, index=False)
    print(f"Predictions saved → {preds_path}")
    curve_path = save_pr_curve(y_test, test_scores, model_name)
    print(f"PR curve saved → {curve_path}")

    return results


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvTran time-series classification baseline")
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
                             f"cosine decay (default: {WARMUP_RATIO}; ConvTran's original paper "
                             f"uses no warmup)")
    parser.add_argument("--emb-size", type=int, default=EMB_SIZE,
                        help=f"ConvTran embedding dimension (default: {EMB_SIZE}, ConvTran's published default)")
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS,
                        help=f"Attention heads (default: {NUM_HEADS}, ConvTran's published default)")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS,
                        help=f"Number of eRPE attention + FFN blocks (default: {N_LAYERS}; "
                             f"published ConvTran is single-block)")
    parser.add_argument("--dim-feedforward", type=int, default=DIM_FEEDFORWARD,
                        help=f"Feed-forward hidden size (default: {DIM_FEEDFORWARD})")
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help=f"Dropout (default: {DROPOUT})")
    parser.add_argument("--model-name", default=None,
                        help="Model label written to result files (default: ConvTran or ConvTran_PR_AUC)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic PyTorch settings for final reproducibility runs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default=os.environ.get("TRAIT_CONVTRAN_DEVICE", "auto"),
                        help="Training device (default: TRAIT_CONVTRAN_DEVICE or auto)")
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
            device_name=args.device, warmup_ratio=args.warmup_ratio,
            emb_size=args.emb_size, num_heads=args.num_heads, n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward, dropout=args.dropout)
