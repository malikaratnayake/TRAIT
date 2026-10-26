#!/usr/bin/env bash
# =============================================================================
# run_pipeline_review.sh — TRAIT Full Pipeline (WACV #491 camera-ready revision)
# =============================================================================
#
# Mirrors run_pipeline.sh end-to-end, with the additions made for the
# camera-ready revision (2026-08):
#
#   * Preprocessing-parity fix (Phase 3): train_fcn.py and train_rocket.py now
#     gap-fill missing detections via the same linear-interpolation-then-SG-
#     smoothing logic used by src/generate_features.py to build the engineered
#     125-feature set (src/utils/smoothing.smooth_coords), instead of dropping
#     missing-detection frames before smoothing. No change needed in this
#     script for that fix — it lives inside train_fcn.py / train_rocket.py, so
#     every call below already gets it.
#
#   * Transformer-family baselines (Phase 3, steps 3.4b/3.4c): two new
#     sequence baselines, both sharing train_fcn.py's preprocessing/scaling/
#     threshold/latency/CI harness so every sequence baseline and the
#     engineered-feature classifiers consume an identical gap-filled,
#     SG-smoothed trajectory.
#       src/train_transformer.py  A compact Transformer encoder (linear
#                projection, learnable positional embedding, standard
#                nn.TransformerEncoder layers, length-masked mean pooling),
#                trained with linear LR warmup + cosine decay (Transformers
#                are warmup-sensitive in a way CNNs are not). Following the
#                general encoder framework used for multivariate time-series
#                representation learning / classification by Zerveas et al.
#                (KDD 2021).
#       src/train_convtran.py     A faithful reimplementation of ConvTran
#                (Foumani et al., DMKD 2024): disjoint-CNN convolutional
#                tokeniser, tAPE positional encoding, and eRPE relative-
#                position attention.
#     Added in response to reviewer requests for modern Transformer-based
#     sequence baselines alongside the existing 1D-CNN and miniROCKET. No
#     graph-based baseline is included: each sample is a single centroid
#     trajectory with no interacting agents and no intrinsic spatial graph
#     topology.
#
#   * ConvTran batch-size sensitivity sweep (Phase 3, step 3.4d) —
#     SUPPLEMENTARY and OPT-IN via --convtran-batch-sweep "16 32 64 ...".
#     ConvTran is trained at its published architecture defaults but at
#     batch_size=128 rather than the official recipe's 16, for parity with the
#     other sequence baselines; this sweep shows the reported result is stable
#     across that choice. Off by default — a normal run is unaffected.
#
#     Note: a no-LR-warmup ablation of the Transformer encoder
#     (--warmup-ratio 0) was also run during development to check that
#     under-optimisation wasn't responsible for its result (PR-AUC 0.8068 vs
#     0.8002 with warmup — statistically indistinguishable, ruling that out).
#     It is not part of this automated pipeline and is not reported in the
#     paper; src/train_transformer.py still accepts --warmup-ratio for anyone
#     who wants to reproduce that check manually.
#
# Phases (identical structure to run_pipeline.sh):
#
#   Phase 1  Feature generation   — Savitzky-Golay smoothing (window=11, poly=2)
#                                   applied to raw coordinates; 125 motion features
#                                   extracted across 6 biological domains.
#                                   Tracks shorter than 5 frames are discarded.
#
#   Phase 2  CV splits            — Station-disjoint grouped 5-fold splits.
#                                   Default: StratifiedGroupKFold.
#                                   Fold 4 is held out as the fixed test set.
#
#   Phase 3  Model performance    — Majority baseline, Logistic Regression,
#                                   Random Forest, XGBoost, 1D-CNN, miniROCKET,
#                                   Transformer, ConvTran.
#                                   Output: CSV comparison table + PR-AUC figure.
#
#   Phase 4  Cross-farm           — Farm A→B and B→A transfer (no adaptation).
#
#   Phase 5  Ablation studies     — Cumulative domain (D1→D1+2→…→D1-6),
#                                   Single-domain-only (D1 only, D2 only, …),
#                                   Leave-one-group-out (32 biological subgroups).
#
# Usage:
#   bash run_pipeline_review.sh --fps 30
#   bash run_pipeline_review.sh --fps 30 --from-phase 3
#   bash run_pipeline_review.sh --fps 30 --run-name my_run
#   bash run_pipeline_review.sh --dry-run --fps 30
#
# Options:
#   --fps FLOAT          Frames per second for feature extraction [required on
#                        first run; cached in data/pipeline.cfg afterwards]
#   --from-phase N       Restart from phase N (1–5)
#   --results-dir PATH   Use an explicit results directory
#   --run-name NAME      Save results under data/results/<NAME>/
#   --overwrite-results  Remove the selected results directory and re-run
#   --tracks-dir PATH    Override default data/tracks/
#   --labels-csv PATH    Override default data/track_classification.csv
#   --features-csv PATH  Override feature CSV path
#   --splits-csv PATH    Override split CSV path
#   --fold-strategy NAME Split strategy: stratified_group (default) or group
#   --defer-sequence-baselines
#                        Run Phase 4/5 before 1D-CNN, miniROCKET, Transformer,
#                        and ConvTran
#   --transformer-batch-size N
#                        Override the Transformer's training/eval batch size
#                        (default: train_transformer.py's own default, 128).
#                        Lower this on memory-constrained machines — self-
#                        attention memory scales with batch_size * max_len^2,
#                        and at max_len=512 a batch size of 128 can exceed a
#                        few GB of RAM. 16–32 is a safe starting point on
#                        constrained hardware.
#   --convtran-batch-size N
#                        Override ConvTran's training/eval batch size (default:
#                        train_convtran.py's own default, 128). Same memory
#                        caveat as --transformer-batch-size applies (ConvTran's
#                        eRPE attention is also O(max_len^2)).
#   --convtran-batch-sweep "N N N"
#                        OPT-IN supplementary study (step 3.4d). Trains ConvTran
#                        once per listed batch size and writes a summary table,
#                        to show the reported ConvTran result is stable across
#                        this choice rather than an artefact of one setting
#                        (batch_size=128 is a deliberate deviation from
#                        ConvTran's published default of 16, chosen for parity
#                        with the other sequence baselines). Pass a quoted,
#                        space-separated list, e.g.
#                          --convtran-batch-sweep "16 32 64 96 128 192"
#                        SLOW: this is a full training run per batch size, and
#                        smaller batch sizes take proportionally longer (more
#                        optimiser steps per epoch). Omitted by default, so it
#                        never affects a normal run. It writes to its own
#                        convtran_batch_size_study/ directory and never touches
#                        the reported ConvTran result in convtran_prauc/.
#   --test-run           Reproducibility mode for final reporting: single-thread
#                        model fitting, deterministic PyTorch, CPU FCN/Transformer/ConvTran.
#   --dry-run            Print commands without executing
#   -h, --help           Show this help and exit
#
# Checkpoints:
#   Each step writes a sentinel file to RESULTS/ckpts/.
#   Re-running the script automatically skips completed steps.
#   Use --overwrite-results or --from-phase N to override.
#
# 1D-CNN, Transformer, and ConvTran (Phase 3) need torch; miniROCKET needs
# aeon — installed into the same environment as everything else, see
# requirements.txt. No new dependencies were introduced for the Transformer
# baselines, ConvTran, or the preprocessing fix beyond what run_pipeline.sh
# already requires (ConvTran's attention is implemented with plain
# reshape/permute rather than einops, to avoid a new dependency).
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO="$SCRIPT_DIR"

# ── Default paths ─────────────────────────────────────────────────────────────
TRACKS_DIR="$REPO/data/tracks"
LABELS_CSV="$REPO/data/track_classification.csv"
SPLITS_CSV="$REPO/data/splits_folds.csv"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-1000}"
RESULTS_ROOT="$REPO/data/results"
RESULTS="$RESULTS_ROOT/trait_results_review"
LOGS_DIR="$REPO/logs"
CFG_FILE="$REPO/data/pipeline.cfg"
PYTHON="${PYTHON:-python3}"
FEATURES_CSV_OVERRIDE=""

# ── Argument defaults ─────────────────────────────────────────────────────────
FPS=""
FROM_PHASE=1
FORCE=0
OVERWRITE_RESULTS=0
RUN_NAME=""
DRY_RUN=0
FOLD_STRATEGY="stratified_group"
DEFER_SEQUENCE_BASELINES=0
TEST_RUN=0
PIPELINE_MODEL_N_JOBS="${PIPELINE_MODEL_N_JOBS:--1}"
TRANSFORMER_BATCH_SIZE="${TRANSFORMER_BATCH_SIZE:-}"
CONVTRAN_BATCH_SIZE="${CONVTRAN_BATCH_SIZE:-}"
CONVTRAN_BATCH_SWEEP="${CONVTRAN_BATCH_SWEEP:-}"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fps)           FPS="$2";           shift 2 ;;
        --from-phase)    FROM_PHASE="$2";    shift 2 ;;
        --results-dir)   RESULTS="$2";       shift 2 ;;
        --run-name)      RUN_NAME="$2";      shift 2 ;;
        --overwrite-results|--overwrite|--force)
                         OVERWRITE_RESULTS=1; FORCE=1; shift ;;
        --tracks-dir)    TRACKS_DIR="$2";    shift 2 ;;
        --labels-csv)    LABELS_CSV="$2";    shift 2 ;;
        --features-csv)  FEATURES_CSV_OVERRIDE="$2"; shift 2 ;;
        --splits-csv)    SPLITS_CSV="$2";    shift 2 ;;
        --fold-strategy) FOLD_STRATEGY="$2"; shift 2 ;;
        --defer-sequence-baselines|--sequence-after-phase5)
                         DEFER_SEQUENCE_BASELINES=1; shift ;;
        --transformer-batch-size)
                         TRANSFORMER_BATCH_SIZE="$2"; shift 2 ;;
        --convtran-batch-size)
                         CONVTRAN_BATCH_SIZE="$2"; shift 2 ;;
        --convtran-batch-sweep)
                         CONVTRAN_BATCH_SWEEP="$2"; shift 2 ;;
        --test-run|--deterministic-run|--final-run)
                         TEST_RUN=1; shift ;;
        --dry-run)       DRY_RUN=1;          shift ;;
        -h|--help)
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 && !/^#/ { exit }' "$0"
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "$FOLD_STRATEGY" in
    stratified_group|group) ;;
    *)
        echo "Unknown fold strategy: $FOLD_STRATEGY (expected stratified_group or group)" >&2
        exit 1 ;;
esac

# ── Resolve run-specific output paths ─────────────────────────────────────────
abs_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then echo "$p"; else echo "$REPO/$p"; fi
}

if [[ -n "$RUN_NAME" ]]; then
    RESULTS="$RESULTS_ROOT/$RUN_NAME"
fi

TRACKS_DIR="$(abs_path "$TRACKS_DIR")"
LABELS_CSV="$(abs_path "$LABELS_CSV")"
SPLITS_CSV="$(abs_path "$SPLITS_CSV")"
RESULTS="$(abs_path "$RESULTS")"
SPLITS_OUTPUT_DIR="$(dirname "$SPLITS_CSV")"
SPLITS_GENERATED_CSV="$SPLITS_OUTPUT_DIR/splits_folds.csv"
CKPTS="$RESULTS/ckpts"
FEATURES_CSV="${FEATURES_CSV_OVERRIDE:-$RESULTS/track_features.csv}"
FEATURES_CSV="$(abs_path "$FEATURES_CSV")"
FEATURE_CKPT_DIR="$RESULTS/feature_ckpts"
FIGURES_DIR="$RESULTS/figures"

# ── Load cached FPS if not provided ──────────────────────────────────────────
if [[ -z "$FPS" && -f "$CFG_FILE" ]]; then
    FPS=$(grep "^fps=" "$CFG_FILE" 2>/dev/null | cut -d= -f2 || true)
fi

# ── Overwrite mode ────────────────────────────────────────────────────────────
if [[ "$OVERWRITE_RESULTS" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    echo "  overwrite: removing $RESULTS …"
    rm -rf "$RESULTS"
fi

# ── Setup logging ─────────────────────────────────────────────────────────────
mkdir -p "$LOGS_DIR"
if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$CKPTS"
fi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/pipeline_review_${TIMESTAMP}.log"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

emit()     { echo -e "$*" | tee -a "$LOG_FILE"; }
emit_err() { echo -e "$*" | tee -a "$LOG_FILE" >&2; }
log()      { emit "${CYAN}[$(date +%H:%M:%S)]${RESET} $*"; }
ok()       { emit "${GREEN}[$(date +%H:%M:%S)] ✓${RESET} $*"; }
warn()     { emit "${YELLOW}[$(date +%H:%M:%S)] ⚠${RESET} $*"; }
err()      { emit_err "${RED}[$(date +%H:%M:%S)] ✗ ERROR:${RESET} $*"; }
header() {
    emit "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"
    emit "${BOLD}${CYAN}  $*${RESET}"
    emit "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"
}

# ── Checkpoint helpers ────────────────────────────────────────────────────────
ckpt_file() { echo "$CKPTS/${1}.done"; }

checkpoint_mode_matches() {
    local ckpt="$1" phase="$2"
    [[ "$TEST_RUN" -eq 1 && "$phase" -ge 3 ]] || return 0
    grep -qx "test_run=1" "$ckpt" 2>/dev/null &&
        grep -qx "model_n_jobs=$PIPELINE_MODEL_N_JOBS" "$ckpt" 2>/dev/null
}

needs_run() {
    local name="$1" phase="$2" ckpt
    [[ "$FORCE" -eq 1 ]] && return 0
    [[ "$phase" -lt "$FROM_PHASE" ]] && return 1
    ckpt="$(ckpt_file "$name")"
    if [[ -f "$ckpt" ]]; then
        if ! checkpoint_mode_matches "$ckpt" "$phase"; then
            log "  [rerun] $name — checkpoint was not produced in test-run mode"
            return 0
        fi
        log "  [skip] $name — checkpoint exists"
        return 1
    fi
    return 0
}

needs_run_with_deps() {
    local name="$1" phase="$2" dep
    shift 2
    [[ "$FORCE" -eq 1 ]] && return 0
    [[ "$phase" -lt "$FROM_PHASE" ]] && return 1

    local ckpt
    ckpt="$(ckpt_file "$name")"
    if [[ ! -f "$ckpt" ]]; then
        return 0
    fi
    if ! checkpoint_mode_matches "$ckpt" "$phase"; then
        log "  [rerun] $name — checkpoint was not produced in test-run mode"
        return 0
    fi
    for dep in "$@"; do
        if [[ -e "$dep" && "$dep" -nt "$ckpt" ]]; then
            log "  [rerun] $name — dependency newer: ${dep#$REPO/}"
            return 0
        fi
    done
    log "  [skip] $name — checkpoint exists"
    return 1
}

mark_done() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  [dry-run] would mark $1 complete"
    else
        {
            echo "test_run=$TEST_RUN"
            echo "model_n_jobs=$PIPELINE_MODEL_N_JOBS"
            echo "timestamp=$(date +%Y-%m-%dT%H:%M:%S%z)"
        } > "$(ckpt_file "$1")"
        ok "$1"
    fi
}

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        local rendered="" arg
        for arg in "$@"; do
            printf -v rendered '%s%q ' "$rendered" "$arg"
        done
        emit "[dry-run] ${rendered% }"
    else
        "$@" 2>&1 | tee -a "$LOG_FILE"
        return "${PIPESTATUS[0]}"
    fi
}

# ── Final-report reproducibility mode ────────────────────────────────────────
if [[ "$TEST_RUN" -eq 1 ]]; then
    PIPELINE_MODEL_N_JOBS=1
    export TRAIT_DETERMINISTIC=1
    export TRAIT_N_JOBS=1
    export TRAIT_FCN_DEVICE=cpu
    export TRAIT_TRANSFORMER_DEVICE=cpu
    export TRAIT_CONVTRAN_DEVICE=cpu
    export PYTHONHASHSEED=42
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    log "  Test-run reproducibility mode: n_jobs=1, deterministic PyTorch, CPU FCN/Transformer/ConvTran."
else
    export TRAIT_N_JOBS="$PIPELINE_MODEL_N_JOBS"
fi
export PIPELINE_TEST_RUN="$TEST_RUN"
export PIPELINE_MODEL_N_JOBS="$PIPELINE_MODEL_N_JOBS"

# ── Verify Python ─────────────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    err "Python not found: $PYTHON  (set PYTHON= env var to override)"
    exit 1
fi
PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
FEATURE_COUNT=$("$PYTHON" - <<'PYEOF'
import re
from pathlib import Path
text = Path("src/models/data_loader.py").read_text()
match = re.search(r"assert len\(ALL_FEATURES\) == (\d+)", text)
print(match.group(1) if match else "125")
PYEOF
)
log "Python $PYVER at $(command -v "$PYTHON")"
log "Log: $LOG_FILE"
log "Results dir : $RESULTS"
log "Features CSV: $FEATURES_CSV"
log "Figures dir : $FIGURES_DIR"
log "Motion feature count: $FEATURE_COUNT"

if "$PYTHON" -c "import xgboost" >/dev/null 2>&1; then
    XGBOOST_AVAILABLE=1
else
    XGBOOST_AVAILABLE=0
fi

export PIPELINE_RESULTS="$RESULTS"
export PIPELINE_FEATURES_CSV="$FEATURES_CSV"
export PIPELINE_SPLITS_CSV="$SPLITS_CSV"
export PIPELINE_FIGURES_DIR="$FIGURES_DIR"
export PIPELINE_TRACKS_DIR="$TRACKS_DIR"
export PIPELINE_FEATURE_COUNT="$FEATURE_COUNT"

# Ensure all src scripts can resolve 'from src.models / src.utils / src.train_fcn' imports
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# =============================================================================
# PHASE 1 — Feature Generation
# =============================================================================
header "Phase 1 — Feature Generation  (smoothing + 125-feature extraction)"
log "  Savitzky-Golay smoothing (window=11, poly=2) is applied to raw (x,y)"
log "  coordinates before feature extraction. Tracks < 5 frames are discarded."
log "  Features: $FEATURE_COUNT intrinsic motion features across 6 biological domains."

# 1.1 Extract six-domain motion feature set
STEP="p1_features"
if needs_run "$STEP" 1; then
    log "1.1  Extracting $FEATURE_COUNT motion features …"
    if [[ -z "$FPS" ]]; then
        err "FPS is required for feature extraction."
        err "Re-run with: bash run_pipeline_review.sh --fps 30"
        exit 1
    fi
    [[ ! -d "$TRACKS_DIR" ]] && { err "Tracks dir not found: $TRACKS_DIR"; exit 1; }
    echo "fps=$FPS" > "$CFG_FILE"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        run_cmd "$PYTHON" src/generate_features.py \
            --tracks-dir "$TRACKS_DIR" \
            --labels-csv "$LABELS_CSV" \
            --fps "$FPS" \
            --output-csv "$FEATURES_CSV" \
            --ckpt-dir "$FEATURE_CKPT_DIR" \
            --batch-size "$FEATURE_BATCH_SIZE"
        run_cmd "$PYTHON" src/generate_features.py \
            --tracks-dir "$TRACKS_DIR" \
            --labels-csv "$LABELS_CSV" \
            --fps "$FPS" \
            --output-csv "$FEATURES_CSV" \
            --ckpt-dir "$FEATURE_CKPT_DIR" \
            --finalize
    else
        mkdir -p "$FEATURE_CKPT_DIR"
        while true; do
            before=$(find "$FEATURE_CKPT_DIR" -maxdepth 1 -name 'batch_*.csv' | wc -l | tr -d ' ')
            batch_output=$("$PYTHON" src/generate_features.py \
                --tracks-dir "$TRACKS_DIR" \
                --labels-csv "$LABELS_CSV" \
                --fps "$FPS" \
                --output-csv "$FEATURES_CSV" \
                --ckpt-dir "$FEATURE_CKPT_DIR" \
                --batch-size "$FEATURE_BATCH_SIZE")
            printf '%s\n' "$batch_output" | tee -a "$LOG_FILE"
            if grep -q "Run --finalize to merge" <<<"$batch_output"; then
                break
            fi
            after=$(find "$FEATURE_CKPT_DIR" -maxdepth 1 -name 'batch_*.csv' | wc -l | tr -d ' ')
            if [[ "$after" -le "$before" ]]; then
                err "Feature extraction made no new checkpoint in $FEATURE_CKPT_DIR"
                exit 1
            fi
        done
        run_cmd "$PYTHON" src/generate_features.py \
            --tracks-dir "$TRACKS_DIR" \
            --labels-csv "$LABELS_CSV" \
            --fps "$FPS" \
            --output-csv "$FEATURES_CSV" \
            --ckpt-dir "$FEATURE_CKPT_DIR" \
            --finalize
    fi
    [[ "$DRY_RUN" -eq 0 && ! -f "$FEATURES_CSV" ]] && {
        err "Feature extraction did not produce $FEATURES_CSV"; exit 1; }
    mark_done "$STEP"
fi

ok "Phase 1 complete"

# =============================================================================
# PHASE 2 — CV Splits
# =============================================================================
header "Phase 2 — Grouped, Leakage-Free CV Splits"
log "  Station-disjoint 5-fold grouped split (group = farm_id + station_id)."
log "  Fold strategy: $FOLD_STRATEGY"
log "  Fold 4 is held out as the fixed test set (not used for model selection)."
log "  Cross-farm roles assigned per fold: Farm A train / Farm B train splits."

split_strategy_matches() {
    [[ -f "$SPLITS_CSV" ]] || return 1
    local current
    current=$("$PYTHON" - "$SPLITS_CSV" "$FOLD_STRATEGY" <<'PYEOF' 2>/dev/null || true
import sys
import pandas as pd

path = sys.argv[1]
expected_strategy = sys.argv[2]
expected_stratification = {
    "stratified_group": "farm_id_class_seed_search_v1",
    "group": "group_v1",
}.get(expected_strategy, "")

try:
    df = pd.read_csv(path, nrows=1)
except Exception:
    print("")
    raise SystemExit(0)

strategy_vals = sorted(
    v for v in df.get("fold_strategy", pd.Series(dtype=str)).dropna().astype(str).unique()
    if v
)
strat_vals = sorted(
    v for v in df.get("fold_stratification", pd.Series(dtype=str)).dropna().astype(str).unique()
    if v
)

if strategy_vals == [expected_strategy] and strat_vals == [expected_stratification]:
    print(expected_strategy)
else:
    print("")
PYEOF
)
    [[ "$current" == "$FOLD_STRATEGY" ]]
}

needs_run_splits() {
    local name="$1" phase="$2"
    [[ "$FORCE" -eq 1 ]] && return 0
    [[ "$phase" -lt "$FROM_PHASE" ]] && return 1
    if [[ ! -f "$(ckpt_file "$name")" ]]; then
        return 0
    fi
    if ! split_strategy_matches; then
        log "  [rerun] $name — fold strategy/metadata differs from existing splits"
        return 0
    fi
    log "  [skip] $name — checkpoint exists"
    return 1
}

# 2.1 Create station-disjoint grouped splits
STEP="p2_splits"
if needs_run_splits "$STEP" 2; then
    log "2.1  Creating station-disjoint grouped splits …"
    [[ ! -f "$LABELS_CSV" ]] && { err "Labels CSV not found: $LABELS_CSV"; exit 1; }
    run_cmd "$PYTHON" src/create_splits.py \
        --labels-csv "$LABELS_CSV" \
        --output-dir "$SPLITS_OUTPUT_DIR" \
        --n-folds 5 \
        --fold-strategy "$FOLD_STRATEGY" \
        --random-state 0 \
        --seed-search-trials 100
    if [[ "$SPLITS_GENERATED_CSV" != "$SPLITS_CSV" ]]; then
        run_cmd cp "$SPLITS_GENERATED_CSV" "$SPLITS_CSV"
    fi
    mark_done "$STEP"
fi

# 2.2 Validate feature / split alignment
STEP="p2_validate"
if needs_run_with_deps "$STEP" 2 "$FEATURES_CSV" "$SPLITS_CSV" "$REPO/run_pipeline_review.sh"; then
    log "2.2  Validating features ↔ splits alignment …"
run_cmd "$PYTHON" - <<'PYEOF'
import os, sys
from pathlib import Path
import pandas as pd

feat = pd.read_csv(os.environ["PIPELINE_FEATURES_CSV"])
spl  = pd.read_csv(os.environ["PIPELINE_SPLITS_CSV"])
missing = set(spl["filename"]) - set(feat["filename"])
if missing:
    sample = ", ".join(sorted(missing)[:5])
    raise SystemExit(
        f"ERROR: {len(missing)} tracks in splits are missing from the features CSV. "
        f"First missing files: {sample}"
    )
else:
    n = len(feat)
    label_col = "label" if "label" in feat.columns else None
    if label_col:
        labels = feat[label_col]
        n_ins = int((labels == 1).sum()) if pd.api.types.is_numeric_dtype(labels) \
                else int(labels.astype(str).str.lower().eq("insect").sum())
    else:
        n_ins = "?"
    folds  = spl["fold"].value_counts().sort_index()
    meta   = {"filename","filepath","label","fold","split_role","cross_farm_role",
              "fold_strategy","fold_random_state",
              "farm_name","farm_id","location_num","station_id","group_key","y"}
    n_feats = sum(1 for c in feat.columns if c not in meta)
    print(f"  Tracks   : {n:,}  (insect: {n_ins}, non-insect: {n - n_ins if isinstance(n_ins,int) else '?'})")
    print(f"  Features : {n_feats}")
    print(f"  Folds    : {dict(folds)}")
    print("  Alignment OK")
PYEOF
    mark_done "$STEP"
fi

ok "Phase 2 complete"

# =============================================================================
# PHASE 3 — Model Performance
# =============================================================================
header "Phase 3 — Model Performance"
log "  Models: Majority baseline · Logistic Regression · Random Forest ·"
log "          XGBoost · 1D-CNN · miniROCKET · Transformer · ConvTran"
log "  Selection metric: validation PR-AUC.  Threshold tuning: validation F1."
log "  Final evaluation: held-out test set (fold 4).  Metric: PR-AUC, F1, Prec, Rec."
log "  1D-CNN / miniROCKET / Transformer / ConvTran all share identical gap-filled,"
log "  SG-smoothed preprocessing (src/utils/smoothing.smooth_coords)."

# 3.1 Majority baseline
STEP="p3_majority"
if needs_run_with_deps "$STEP" 3 "$FEATURES_CSV" "$SPLITS_CSV" \
    "$REPO/src/models/majority_baseline.py" "$REPO/run_pipeline_review.sh"; then
    log "3.1  Training majority baseline …"
    run_cmd "$PYTHON" src/models/majority_baseline.py \
        --features-csv "$FEATURES_CSV" \
        --splits-csv   "$SPLITS_CSV" \
        --output-dir   "$RESULTS" \
        --models majority
    mark_done "$STEP"
fi

# 3.2 LR / RF / XGBoost with PR-AUC model selection
STEP="p3_tabular"
if needs_run_with_deps "$STEP" 3 "$FEATURES_CSV" "$SPLITS_CSV" \
    "$REPO/src/models/tabular_models.py" "$REPO/run_pipeline_review.sh"; then
    TABULAR_MODELS=(lr rf)
    if [[ "$XGBOOST_AVAILABLE" -eq 1 ]]; then
        TABULAR_MODELS+=(xgb)
        log "3.2  Training Logistic Regression, Random Forest, XGBoost (PR-AUC selection) …"
    else
        log "3.2  Training Logistic Regression and Random Forest (PR-AUC selection) …"
        warn "xgboost not found in '$PYTHON' — skipping XGBoost, Phase 4, and Phase 5"
        warn "To enable: $PYTHON -m pip install -r requirements.txt"
    fi
    run_cmd "$PYTHON" src/models/tabular_models.py \
        --features-csv "$FEATURES_CSV" \
        --splits-csv   "$SPLITS_CSV" \
        --output-dir   "$RESULTS" \
        --n-jobs       "$PIPELINE_MODEL_N_JOBS" \
        --models "${TABULAR_MODELS[@]}"
    mark_done "$STEP"
fi

run_phase3_sequence_baselines() {
    # 3.3 1D-CNN with PR-AUC early stopping
    STEP="p3_fcn"
    if needs_run_with_deps "$STEP" 3 "$SPLITS_CSV" \
        "$REPO/src/train_fcn.py" "$REPO/src/utils/smoothing.py" "$REPO/run_pipeline_review.sh"; then
        log "3.3  Training 1D-CNN with PR-AUC early stopping …"
        if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
            warn "PyTorch not found in '$PYTHON' — skipping 1D-CNN"
            warn "To enable: $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
            mark_done "$STEP"
        else
            FCN_ARGS=(
                --max_len 512
                --early-stop-metric pr_auc
                --model-name FCN_PR_AUC
                --output-dir "$RESULTS/fcn_prauc"
            )
            if [[ "$TEST_RUN" -eq 1 ]]; then
                FCN_ARGS+=(--deterministic --device cpu)
            fi
            run_cmd "$PYTHON" src/train_fcn.py "${FCN_ARGS[@]}"
            mark_done "$STEP"
        fi
    fi

    # 3.4 miniROCKET with PR-AUC ridge selection
    STEP="p3_rocket"
    if needs_run_with_deps "$STEP" 3 "$SPLITS_CSV" \
        "$REPO/src/train_rocket.py" "$REPO/src/utils/smoothing.py" "$REPO/run_pipeline_review.sh"; then
        log "3.4  Training miniROCKET with PR-AUC ridge selection …"
        if ! "$PYTHON" -c "import aeon" >/dev/null 2>&1; then
            warn "aeon not found in '$PYTHON' — skipping miniROCKET"
            warn "To enable: $PYTHON -m pip install -r requirements.txt"
            mark_done "$STEP"
        else
            run_cmd "$PYTHON" src/train_rocket.py \
                --max_len 512 \
                --n-jobs "$PIPELINE_MODEL_N_JOBS" \
                --selection-metric pr_auc \
                --model-name MiniROCKET_PR_AUC \
                --output-dir "$RESULTS/rocket_prauc"
            mark_done "$STEP"
        fi
    fi

    # 3.4b Transformer with LR warmup + PR-AUC early stopping (camera-ready
    # revision addition). A no-warmup ablation was also run during
    # development (see header comment) but is not part of this pipeline.
    STEP="p3_transformer_warmup"
    if needs_run_with_deps "$STEP" 3 "$SPLITS_CSV" \
        "$REPO/src/train_transformer.py" "$REPO/src/train_fcn.py" "$REPO/run_pipeline_review.sh"; then
        log "3.4b  Training Transformer with PR-AUC early stopping …"
        if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
            warn "PyTorch not found in '$PYTHON' — skipping Transformer"
            warn "To enable: $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
            mark_done "$STEP"
        else
            TRANSFORMER_ARGS=(
                --max_len 512
                --early-stop-metric pr_auc
                --model-name Transformer_Warmup_PR_AUC
                --output-dir "$RESULTS/transformer_warmup_prauc"
            )
            if [[ -n "$TRANSFORMER_BATCH_SIZE" ]]; then
                TRANSFORMER_ARGS+=(--batch-size "$TRANSFORMER_BATCH_SIZE")
            fi
            if [[ "$TEST_RUN" -eq 1 ]]; then
                TRANSFORMER_ARGS+=(--deterministic --device cpu)
            fi
            log "  Note: self-attention memory scales with batch_size * max_len^2."
            log "  If this step is killed (OOM) on constrained hardware, re-run with"
            log "  --transformer-batch-size 16 (or 32)."
            run_cmd "$PYTHON" src/train_transformer.py "${TRANSFORMER_ARGS[@]}"
            mark_done "$STEP"
        fi
    fi

    # 3.4c ConvTran (camera-ready revision addition)
    STEP="p3_convtran"
    if needs_run_with_deps "$STEP" 3 "$SPLITS_CSV" \
        "$REPO/src/train_convtran.py" "$REPO/src/train_transformer.py" "$REPO/src/train_fcn.py" \
        "$REPO/run_pipeline_review.sh"; then
        log "3.4c  Training ConvTran (Foumani et al., DMKD 2024) with PR-AUC early stopping …"
        if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
            warn "PyTorch not found in '$PYTHON' — skipping ConvTran"
            warn "To enable: $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
            mark_done "$STEP"
        else
            CONVTRAN_ARGS=(
                --max_len 512
                --early-stop-metric pr_auc
                --model-name ConvTran_PR_AUC
                --output-dir "$RESULTS/convtran_prauc"
            )
            if [[ -n "$CONVTRAN_BATCH_SIZE" ]]; then
                CONVTRAN_ARGS+=(--batch-size "$CONVTRAN_BATCH_SIZE")
            fi
            if [[ "$TEST_RUN" -eq 1 ]]; then
                CONVTRAN_ARGS+=(--deterministic --device cpu)
            fi
            log "  Note: eRPE attention memory also scales with batch_size * max_len^2."
            log "  If this step is killed (OOM) on constrained hardware, re-run with"
            log "  --convtran-batch-size 16 (or 32)."
            run_cmd "$PYTHON" src/train_convtran.py "${CONVTRAN_ARGS[@]}"
            mark_done "$STEP"
        fi
    fi

    # 3.4d ConvTran batch-size sensitivity sweep — SUPPLEMENTARY, opt-in.
    # Only runs when --convtran-batch-sweep is given, so a normal pipeline run
    # is unaffected. Delegates to src/convtran_batch_size_study.py (which
    # handles per-batch-size output dirs, resume, per-run failure isolation,
    # and aggregation) rather than reimplementing that loop here. Writes to
    # $RESULTS/convtran_batch_size_study/ and never touches the reported
    # ConvTran result in $RESULTS/convtran_prauc/.
    if [[ -n "$CONVTRAN_BATCH_SWEEP" ]]; then
        STEP="p3_convtran_sweep"
        if needs_run_with_deps "$STEP" 3 "$SPLITS_CSV" \
            "$REPO/src/train_convtran.py" "$REPO/src/convtran_batch_size_study.py" \
            "$REPO/run_pipeline_review.sh"; then
            log "3.4d  ConvTran batch-size sweep (supplementary): [$CONVTRAN_BATCH_SWEEP] …"
            if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
                warn "PyTorch not found in '$PYTHON' — skipping ConvTran batch-size sweep"
                warn "To enable: $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
                mark_done "$STEP"
            else
                # IFS is $'\n\t' script-wide, so an unquoted expansion would NOT
                # split on spaces. Split explicitly with IFS=' ' for this read.
                IFS=' ' read -r -a CONVTRAN_SWEEP_SIZES <<< "$CONVTRAN_BATCH_SWEEP"
                SWEEP_ARGS=(
                    --batch-sizes "${CONVTRAN_SWEEP_SIZES[@]}"
                    --max_len 512
                    --early-stop-metric pr_auc
                    --output-dir "$RESULTS"
                    --python "$PYTHON"
                )
                if [[ "$TEST_RUN" -eq 1 ]]; then
                    SWEEP_ARGS+=(--device cpu)
                fi
                log "  This trains ConvTran once per batch size — expect it to be SLOW."
                log "  Smaller batch sizes take proportionally longer (more steps/epoch)."
                log "  Already-completed batch sizes are skipped on re-run."
                log "  Output: $RESULTS/convtran_batch_size_study/"
                run_cmd "$PYTHON" src/convtran_batch_size_study.py "${SWEEP_ARGS[@]}"
                mark_done "$STEP"
            fi
        fi
    fi
}

run_phase3_summary_outputs() {
    # 3.5 Comparison table (CSV)
    STEP="p3_compare"
    if needs_run_with_deps "$STEP" 3 \
        "$RESULTS/trivial/trivial_results.json" \
        "$RESULTS/prauc_protocol/prauc_protocol_comparison.csv" \
        "$RESULTS/fcn_prauc/fcn_results.csv" \
        "$RESULTS/rocket_prauc/rocket_results.csv" \
        "$RESULTS/transformer_warmup_prauc/transformer_results.csv" \
        "$RESULTS/convtran_prauc/convtran_results.csv" \
        "$REPO/run_pipeline_review.sh"; then
        log "3.5  Building model comparison table …"
	run_cmd "$PYTHON" - <<'PYEOF'
import json, math, sys, os
from pathlib import Path
import pandas as pd

R = Path(os.environ["PIPELINE_RESULTS"])

def nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))

def fmt(v, spec=".4f"):
    return f"{v:{spec}}" if not nan(v) else "   —  "

def read_model(path, model_key=None):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if model_key:
        d = d.get(model_key, {})
    m   = d.get("test_metrics", d.get("metrics", {}))
    ci  = d.get("bootstrap_ci", d.get("ci", {}))
    lat = d.get("latency", {})
    if not m:
        return None
    f1_ci = ci.get("f1", {}) if isinstance(ci.get("f1"), dict) else {}
    pr_ci = ci.get("pr_auc", {}) if isinstance(ci.get("pr_auc"), dict) else {}
    return {
        "precision":       m.get("precision"),
        "recall":          m.get("recall"),
        "f1":              m.get("f1"),
        "f1_ci_lo":        f1_ci.get("lo"),
        "f1_ci_hi":        f1_ci.get("hi"),
        "pr_auc":          m.get("pr_auc"),
        "pr_auc_ci_lo":    pr_ci.get("lo"),
        "pr_auc_ci_hi":    pr_ci.get("hi"),
        "latency_mean_ms": lat.get("latency_mean_ms"),
    }

rows = []

# Majority baseline
r = read_model(R / "trivial" / "trivial_results.json", "majority")
if r:
    rows.append({"Model": "Majority baseline", "n_features": "—", **r})

# Tabular models (LR, RF, XGBoost) via PR-AUC selection protocol
prauc_csv = R / "prauc_protocol" / "prauc_protocol_comparison.csv"
if prauc_csv.exists():
    pdf = pd.read_csv(prauc_csv)
    for _, row in pdf.iterrows():
        model = str(row.get("model", ""))
        slug  = str(row.get("slug", ""))
        if "Generic traj." in model or slug.startswith("generic_trajectory"):
            continue
        rows.append({
            "Model":           model,
            "n_features":      row.get("n_features", "125"),
            "precision":       row.get("precision"),
            "recall":          row.get("recall"),
            "f1":              row.get("f1"),
            "f1_ci_lo":        row.get("f1_ci_lo"),
            "f1_ci_hi":        row.get("f1_ci_hi"),
            "pr_auc":          row.get("pr_auc"),
            "pr_auc_ci_lo":    row.get("pr_auc_ci_lo"),
            "pr_auc_ci_hi":    row.get("pr_auc_ci_hi"),
            "latency_mean_ms": row.get("latency_mean_ms"),
        })

# 1D-CNN
fcn_csv = R / "fcn_prauc" / "fcn_results.csv"
if fcn_csv.exists():
    fdf = pd.read_csv(fcn_csv)
    for _, row in fdf.iterrows():
        rows.append({
            "Model":           "1D-CNN",
            "n_features":      "—",
            "precision":       row.get("precision"),
            "recall":          row.get("recall"),
            "f1":              row.get("f1"),
            "f1_ci_lo":        row.get("f1_lo"),
            "f1_ci_hi":        row.get("f1_hi"),
            "pr_auc":          row.get("pr_auc"),
            "pr_auc_ci_lo":    row.get("pr_auc_lo"),
            "pr_auc_ci_hi":    row.get("pr_auc_hi"),
            "latency_mean_ms": row.get("latency_mean_ms"),
        })

# miniROCKET
rocket_csv = R / "rocket_prauc" / "rocket_results.csv"
if rocket_csv.exists():
    rdf = pd.read_csv(rocket_csv)
    for _, row in rdf.iterrows():
        rows.append({
            "Model":           "miniROCKET",
            "n_features":      "—",
            "precision":       row.get("precision"),
            "recall":          row.get("recall"),
            "f1":              row.get("f1"),
            "f1_ci_lo":        row.get("f1_lo"),
            "f1_ci_hi":        row.get("f1_hi"),
            "pr_auc":          row.get("pr_auc"),
            "pr_auc_ci_lo":    row.get("pr_auc_lo"),
            "pr_auc_ci_hi":    row.get("pr_auc_hi"),
            "latency_mean_ms": row.get("latency_mean_ms"),
        })

# Transformer-family baselines (camera-ready revision addition)
transformer_stages = [
    ("Transformer", R / "transformer_warmup_prauc" / "transformer_results.csv"),
    ("ConvTran",    R / "convtran_prauc" / "convtran_results.csv"),
]
for label, csv_path in transformer_stages:
    if csv_path.exists():
        tdf = pd.read_csv(csv_path)
        for _, row in tdf.iterrows():
            rows.append({
                "Model":           label,
                "n_features":      "—",
                "precision":       row.get("precision"),
                "recall":          row.get("recall"),
                "f1":              row.get("f1"),
                "f1_ci_lo":        row.get("f1_lo"),
                "f1_ci_hi":        row.get("f1_hi"),
                "pr_auc":          row.get("pr_auc"),
                "pr_auc_ci_lo":    row.get("pr_auc_lo"),
                "pr_auc_ci_hi":    row.get("pr_auc_hi"),
                "latency_mean_ms": row.get("latency_mean_ms"),
            })

if not rows:
    print("  No Phase 3 results found yet."); sys.exit(0)

df = pd.DataFrame(rows)
out = Path(os.environ["PIPELINE_RESULTS"]) / "phase3_model_comparison.csv"
df.to_csv(out, index=False)

# PR-AUC is the paper's primary, threshold-independent metric (Table 3:
# "95% confidence intervals (CI) for PR-AUC were computed using 1000
# bootstrap resamples of the test set"). The CI column below reports the
# PR-AUC CI, not the F1 CI, to match that convention. f1_ci_lo/f1_ci_hi are
# still written to the saved CSV for anyone who wants the F1 interval too.
print(f"\n  {'Model':<26} {'#Feat':>6} {'PR-AUC':>7}  "
      f"{'95% CI (PR-AUC)':^20}  {'F1':>7} {'Rec':>7} {'Prec':>7} {'ms/samp':>8}")
print("  " + "─" * 96)
for _, r in df.iterrows():
    ci = (f"[{fmt(r['pr_auc_ci_lo'])}–{fmt(r['pr_auc_ci_hi'])}]"
          if not nan(r.get("pr_auc_ci_lo")) else "          —         ")
    print(f"  {str(r['Model']):<26} {str(r['n_features']):>6} "
          f"{fmt(r['pr_auc']):>7}  {ci:^20}  "
          f"{fmt(r['f1']):>7} {fmt(r['recall']):>7} {fmt(r['precision']):>7} "
          f"{fmt(r.get('latency_mean_ms'),'.3f'):>8}")
print(f"\n  Saved → {out}")
PYEOF
        mark_done "$STEP"
    fi

    # 3.6 Precision-recall curve figure
    STEP="p3_pr_curves"
    if needs_run_with_deps "$STEP" 3 \
        "$RESULTS/trivial/majority_pr_curve.csv" \
        "$RESULTS/prauc_protocol/logistic_regression_prauc/logistic_regression_prauc_pr_curve.csv" \
        "$RESULTS/prauc_protocol/random_forest_prauc/random_forest_prauc_pr_curve.csv" \
        "$RESULTS/prauc_protocol/xgboost_prauc/xgboost_prauc_pr_curve.csv" \
        "$RESULTS/fcn_prauc/fcn_pr_auc_pr_curve.csv" \
        "$RESULTS/transformer_warmup_prauc/transformer_warmup_pr_auc_pr_curve.csv" \
        "$RESULTS/convtran_prauc/convtran_pr_auc_pr_curve.csv" \
        "$REPO/src/figures/plot_pr_curves.py" \
        "$REPO/run_pipeline_review.sh"; then
        log "3.6  Plotting precision-recall curves (majority, LR, RF, XGBoost, 1D-CNN, miniROCKET, Transformer, ConvTran) …"
        mkdir -p "$FIGURES_DIR"
        run_cmd "$PYTHON" src/figures/plot_pr_curves.py \
            --results-dir "$RESULTS" \
            --output-csv  "$RESULTS/phase3_precision_recall_curves.csv" \
            --output-png  "$FIGURES_DIR/phase3_precision_recall_curves.png"
        mark_done "$STEP"
    fi
}

if [[ "$DEFER_SEQUENCE_BASELINES" -eq 1 ]]; then
    log "  [defer] 1D-CNN, miniROCKET, Transformer, ConvTran, and Phase 3 summaries will run after Phase 5"
    ok "Phase 3 tabular/core complete"
else
    run_phase3_sequence_baselines
    run_phase3_summary_outputs
    ok "Phase 3 complete"
fi

# =============================================================================
# PHASE 4 — Cross-Farm Generalisation
# =============================================================================
header "Phase 4 — Cross-Farm Generalisation"
log "  XGBoost trained on ALL tracks from one farm, evaluated on the other."
log "  No fine-tuning, no domain adaptation, no shared normalisation."
log "  Feature set: $FEATURE_COUNT intrinsic motion features (no recording metadata)."

STEP="p4_cross_farm"
if needs_run_with_deps "$STEP" 4 \
    "$FEATURES_CSV" \
    "$SPLITS_CSV" \
    "$RESULTS/prauc_protocol/xgboost_prauc/results.json" \
    "$REPO/src/cross_farm.py" \
    "$REPO/run_pipeline_review.sh"; then
    XGB_JSON="$RESULTS/prauc_protocol/xgboost_prauc/results.json"
    if [[ "$XGBOOST_AVAILABLE" -eq 0 ]]; then
        warn "xgboost not found in '$PYTHON' — skipping cross-farm XGBoost"
        mark_done "$STEP"
    elif [[ "$DRY_RUN" -eq 0 && ! -f "$XGB_JSON" ]]; then
        warn "Phase 3 XGBoost result missing — skipping cross-farm XGBoost"
        warn "Run with xgboost installed and --from-phase 3 to generate it."
        mark_done "$STEP"
    else
        log "4.1  Running Farm A→B and B→A transfer …"
        if [[ "$DRY_RUN" -eq 1 && ! -f "$XGB_JSON" ]]; then
        warn "Phase 3 XGBoost result missing in dry-run — using placeholder params"
        XGB_PARAMS='{}'
        XGB_THRESHOLD='0.5'
        else
            XGB_PARAMS=$("$PYTHON" -c "
import json
d = json.load(open('$XGB_JSON'))
p = dict(d.get('best_params', {}))
if d.get('selected_n_estimators') is not None:
    p['n_estimators'] = int(d['selected_n_estimators'])
print(json.dumps(p))
" 2>/dev/null || echo '')
            [[ -n "$XGB_PARAMS" && "$XGB_PARAMS" != '{}' ]] || \
                { err "Could not parse XGBoost params from $XGB_JSON"; exit 1; }
            XGB_THRESHOLD=$("$PYTHON" -c "
import json
d = json.load(open('$XGB_JSON'))
print(float(d.get('threshold', 0.5)))
" 2>/dev/null || echo '')
            [[ -n "$XGB_THRESHOLD" ]] || \
                { err "Could not parse XGBoost threshold from $XGB_JSON"; exit 1; }
        fi
        log "  XGB params: $XGB_PARAMS"
        log "  XGB threshold: $XGB_THRESHOLD"
        run_cmd "$PYTHON" src/cross_farm.py \
            --features-csv "$FEATURES_CSV" \
            --splits-csv   "$SPLITS_CSV" \
            --output-dir   "$RESULTS" \
            --xgb-params   "$XGB_PARAMS" \
            --n-jobs       "$PIPELINE_MODEL_N_JOBS" \
            --xgb-threshold "$XGB_THRESHOLD"
        mark_done "$STEP"
    fi
fi

# 4.2 Generalisation gap summary
STEP="p4_gap"
if needs_run_with_deps "$STEP" 4 "$RESULTS/cross_farm/cross_farm_results.csv" "$REPO/run_pipeline_review.sh"; then
    log "4.2  Generalisation gap summary …"
run_cmd "$PYTHON" - <<'PYEOF'
import math, os
import pandas as pd
from pathlib import Path

p = Path(os.environ["PIPELINE_RESULTS"]) / "cross_farm" / "cross_farm_results.csv"
if not p.exists():
    print("  cross_farm_results.csv not found — skipping"); exit(0)

df = pd.read_csv(p)

def fmt(v):
    return f"{v:.4f}" if v is not None and not math.isnan(float(v)) else "  —  "

print(f"\n  {'Experiment':<28} {'PR-AUC':>7}  {'F1':>7}  {'Gap':>10}")
print("  " + "─" * 62)
for _, row in df.iterrows():
    gap_str = ""
    if "to" in str(row.get("evaluation", "")):
        src = str(row.get("train", ""))
        within = df[df["evaluation"] == f"within_{src}"]["pr_auc"]
        if not within.empty and within.iloc[0] is not None:
            g = float(row.get("pr_auc", 0)) - float(within.iloc[0])
            flag = " ✓ GOOD" if abs(g) < 0.05 else (" ⚠ MOD" if abs(g) < 0.10 else " ✗ LARGE")
            gap_str = f"  {g:+.4f}{flag}"
    print(f"  {str(row.get('evaluation','')):<28} {fmt(row.get('pr_auc')):>7}  "
          f"{fmt(row.get('f1')):>7}{gap_str}")
PYEOF
    mark_done "$STEP"
fi

ok "Phase 4 complete"

# =============================================================================
# PHASE 5 — Ablation Studies
# =============================================================================
header "Phase 5 — Ablation Studies  (XGBoost on held-out test set)"
log "  5a  Cumulative domain   D1 → D1+D2 → D1+D2+D3 → … → D1–D6"
log "  5b  Single-domain-only  D1 only, D2 only, …, D6 only"
log "  5c  Leave-one-group-out biological subgroup LOO (32 groups)"

STEP="p5_ablation"
if needs_run_with_deps "$STEP" 5 \
    "$FEATURES_CSV" \
    "$SPLITS_CSV" \
    "$RESULTS/prauc_protocol/xgboost_prauc/results.json" \
    "$REPO/src/ablation.py" \
    "$REPO/run_pipeline_review.sh"; then
    XGB_JSON="$RESULTS/prauc_protocol/xgboost_prauc/results.json"
    if [[ "$XGBOOST_AVAILABLE" -eq 0 ]]; then
        warn "xgboost not found in '$PYTHON' — skipping XGBoost ablation studies"
        mark_done "$STEP"
    elif [[ "$DRY_RUN" -eq 0 && ! -f "$XGB_JSON" ]]; then
        warn "Phase 3 XGBoost result missing — skipping XGBoost ablation studies"
        warn "Run with xgboost installed and --from-phase 3 to generate it."
        mark_done "$STEP"
    else
        if [[ "$DRY_RUN" -eq 1 && ! -f "$XGB_JSON" ]]; then
        warn "Phase 3 XGBoost result missing in dry-run — using placeholder params"
        XGB_PARAMS='{}'
        else
            XGB_PARAMS=$("$PYTHON" -c "
import json
d = json.load(open('$XGB_JSON'))
p = dict(d.get('best_params', {}))
if d.get('selected_n_estimators') is not None:
    p['n_estimators'] = int(d['selected_n_estimators'])
print(json.dumps(p))
" 2>/dev/null || echo '')
            [[ -n "$XGB_PARAMS" && "$XGB_PARAMS" != '{}' ]] || \
                { err "Could not parse XGBoost params from $XGB_JSON"; exit 1; }
        fi
        log "  XGB params: $XGB_PARAMS"
        run_cmd "$PYTHON" src/ablation.py \
            --features-csv "$FEATURES_CSV" \
            --splits-csv   "$SPLITS_CSV" \
            --output-dir   "$RESULTS" \
            --n-jobs       "$PIPELINE_MODEL_N_JOBS" \
            --xgb-params   "$XGB_PARAMS"
        mark_done "$STEP"
    fi
fi

# 5.2 Leave-one-group-out ablation plot
STEP="p5_ablation_plot"
if needs_run_with_deps "$STEP" 5 \
    "$RESULTS/ablation/group_loo_ablation.csv" \
    "$REPO/src/figures/plot_group_loo.py" \
    "$REPO/run_pipeline_review.sh"; then
    if [[ "$DRY_RUN" -eq 0 && ! -f "$RESULTS/ablation/group_loo_ablation.csv" ]]; then
        warn "Group LOO ablation results missing — skipping ablation plots"
    else
        log "5.2  Plotting group leave-one-out ablation effects …"
        mkdir -p "$FIGURES_DIR"
        run_cmd "$PYTHON" src/figures/plot_group_loo.py \
            --input-csv  "$RESULTS/ablation/group_loo_ablation.csv" \
            --output-csv "$RESULTS/ablation/group_loo_top_positive_negative.csv" \
            --output-png "$FIGURES_DIR/group_loo_top_positive_negative.png" \
            || warn "Group LOO F1 plot failed — continuing"
        run_cmd "$PYTHON" src/figures/plot_group_loo.py \
            --input-csv  "$RESULTS/ablation/group_loo_ablation.csv" \
            --output-csv "$RESULTS/ablation/group_loo_prauc_top_positive_negative.csv" \
            --output-png "$FIGURES_DIR/group_loo_prauc_top_positive_negative.png" \
            --metric pr_auc \
            || warn "Group LOO PR-AUC plot failed — continuing"
    fi
    mark_done "$STEP"
fi

# 5.3 Trajectory figures (class overlays + best-session spaghetti)
STEP="p5_trajectory_figs"
if needs_run_with_deps "$STEP" 5 \
    "$SPLITS_CSV" \
    "$REPO/src/figures/plot_class_trajectory_overlays.py" \
    "$REPO/src/figures/plot_session_trajectories.py" \
    "$REPO/src/figures/_track_utils.py" \
    "$REPO/run_pipeline_review.sh"; then
    log "5.3  Plotting trajectory figures …"
    mkdir -p "$FIGURES_DIR"
    run_cmd "$PYTHON" src/figures/plot_class_trajectory_overlays.py \
        --splits-csv "$SPLITS_CSV" \
        --tracks-dir "$TRACKS_DIR" \
        --output-dir "$FIGURES_DIR" \
        || warn "Class-overlay figure failed — continuing"
    run_cmd "$PYTHON" src/figures/plot_session_trajectories.py \
        --splits-csv "$SPLITS_CSV" \
        --tracks-dir "$TRACKS_DIR" \
        --output-dir "$FIGURES_DIR" \
        --rank 1 \
        || warn "Session trajectory figure failed — continuing"
    mark_done "$STEP"
fi

ok "Phase 5 complete"

if [[ "$DEFER_SEQUENCE_BASELINES" -eq 1 ]]; then
    header "Deferred Phase 3 — Sequence Baselines"
    log "  Running deferred 1D-CNN, miniROCKET, Transformer, and ConvTran after Phases 4/5."
    run_phase3_sequence_baselines
    run_phase3_summary_outputs
    ok "Deferred Phase 3 sequence baselines complete"
fi

# =============================================================================
# PIPELINE COMPLETE
# =============================================================================
echo ""
emit "${BOLD}${GREEN}══════════════════════════════════════════════════════════${RESET}"
emit "${BOLD}${GREEN}  Pipeline complete! (camera-ready revision: preprocessing fix + Transformer + ConvTran)${RESET}"
emit "${BOLD}${GREEN}══════════════════════════════════════════════════════════${RESET}"
echo ""
log "Results dir : $RESULTS"
log "Log file    : $LOG_FILE"
echo ""
log "Key output files:"
log "  Phase 1  features       → $FEATURES_CSV"
log "  Phase 2  splits         → $SPLITS_CSV"
log "  Phase 3  comparison     → $RESULTS/phase3_model_comparison.csv"
log "           PR-AUC fig     → $FIGURES_DIR/phase3_precision_recall_curves.png"
log "           1D-CNN         → $RESULTS/fcn_prauc/fcn_results.csv"
log "           miniROCKET     → $RESULTS/rocket_prauc/rocket_results.csv"
log "           Transformer     → $RESULTS/transformer_warmup_prauc/transformer_results.csv"
log "           ConvTran        → $RESULTS/convtran_prauc/convtran_results.csv"
if [[ -n "$CONVTRAN_BATCH_SWEEP" ]]; then
log "           ConvTran sweep  → $RESULTS/convtran_batch_size_study/convtran_batch_size_summary.csv"
fi
log "  Phase 4  cross-farm     → $RESULTS/cross_farm/cross_farm_results.csv"
log "  Phase 5  ablation       → $RESULTS/ablation/"
log "           cumulative     → $RESULTS/ablation/domain_ablation_results.csv"
log "           single-dom     → $RESULTS/ablation/domain_only_results.csv"
log "           group LOO      → $RESULTS/ablation/group_loo_ablation.csv"
log "           LOO figure     → $FIGURES_DIR/group_loo_top_positive_negative.png"
echo ""
