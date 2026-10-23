#!/usr/bin/env bash
# =============================================================================
# run_convtran_batch_sweep.sh — ConvTran batch-size sweep via the full pipeline
# =============================================================================
#
# Runs run_pipeline_review.sh once per ConvTran batch size, archiving each
# batch size's ConvTran results into its own directory, then aggregates them
# into one supplementary summary table.
#
# WHY THIS WRAPPER EXISTS
# -----------------------
# run_pipeline_review.sh accepts --convtran-batch-size, but it is not built
# for sweeps, and two of its behaviours will silently give you wrong results
# if you just call it in a loop yourself:
#
#   1. Checkpoints do not track batch size. mark_done() records only
#      test_run and model_n_jobs, so re-running with a different
#      --convtran-batch-size finds the existing p3_convtran.done checkpoint,
#      sees no newer dependency, and SKIPS the ConvTran step entirely — you
#      would silently get the previous batch size's numbers back. This script
#      deletes that one checkpoint before each iteration.
#
#   2. Every run writes to the same $RESULTS/convtran_prauc/. Successive runs
#      overwrite each other. This script moves each result to
#      $RESULTS/convtran_batch_sweep/convtran_bs<N>/ after it completes.
#
# It also deletes the p3_compare and p3_pr_curves checkpoints each iteration so
# the comparison table and PR-curve figure are regenerated against the batch
# size that just ran, rather than going stale.
#
# WHAT IS AND IS NOT RE-RUN
# -------------------------
# With --from-phase 3 (the default here) and an existing results directory,
# every other step — majority baseline, LR/RF/XGBoost, 1D-CNN, miniROCKET,
# Transformer, Phase 4 cross-farm, Phase 5 ablations — is skipped via its own
# intact checkpoint. Only ConvTran retrains per iteration, plus the two cheap
# summary steps. If you point this at a fresh --run-name with no checkpoints,
# the FIRST iteration will run the entire pipeline from Phase 3 (hours), and
# subsequent iterations will then reuse those checkpoints.
#
# THE PRODUCTION RESULT IS PRESERVED
# ----------------------------------
# $RESULTS/convtran_prauc/ (your reported ConvTran row) is backed up before
# the sweep starts and restored afterwards, so the main comparison table and
# Table 3 numbers are unchanged by running this.
#
# Usage:
#   bash run_convtran_batch_sweep.sh --batch-sizes 16 32 64 96 128 192
#   bash run_convtran_batch_sweep.sh --batch-sizes 16 32 --run-name my_run
#   bash run_convtran_batch_sweep.sh --batch-sizes 16 32 --dry-run
#
# Options:
#   --batch-sizes N [N ...]  Batch sizes to sweep (required)
#   --run-name NAME          Results dir under data/results/ (default:
#                            trait_results_review, matching run_pipeline_review.sh)
#   --fps FLOAT              Passed through to the pipeline (default: cached value)
#   --features-csv PATH      Passed through to the pipeline
#   --from-phase N           Passed through (default: 3)
#   --test-run               Passed through (deterministic / single-threaded / CPU)
#   --dry-run                Print what would happen without running anything
#   -h, --help               Show this help and exit
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO="$SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"

BATCH_SIZES=()
RUN_NAME="trait_results_review"
FPS=""
FEATURES_CSV=""
FROM_PHASE=3
TEST_RUN=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-sizes)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do BATCH_SIZES+=("$1"); shift; done ;;
        --run-name)      RUN_NAME="$2";     shift 2 ;;
        --fps)           FPS="$2";          shift 2 ;;
        --features-csv)  FEATURES_CSV="$2"; shift 2 ;;
        --from-phase)    FROM_PHASE="$2";   shift 2 ;;
        --test-run)      TEST_RUN=1;        shift ;;
        --dry-run)       DRY_RUN=1;         shift ;;
        -h|--help)
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 && !/^#/ { exit }' "$0"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#BATCH_SIZES[@]} -eq 0 ]]; then
    echo "ERROR: --batch-sizes is required (e.g. --batch-sizes 16 32 64 96 128 192)" >&2
    exit 1
fi

RESULTS="$REPO/data/results/$RUN_NAME"
CKPTS="$RESULTS/ckpts"
SWEEP_DIR="$RESULTS/convtran_batch_sweep"
PROD_CONVTRAN="$RESULTS/convtran_prauc"
PROD_BACKUP="$RESULTS/.convtran_prauc_production_backup"

echo "══════════════════════════════════════════════════════════"
echo "  ConvTran batch-size sweep (supplementary study)"
echo "══════════════════════════════════════════════════════════"
echo "  Batch sizes : ${BATCH_SIZES[*]}"
echo "  Results dir : $RESULTS"
echo "  Sweep output: $SWEEP_DIR"
echo "  from-phase  : $FROM_PHASE   test-run: $TEST_RUN"
echo ""

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$SWEEP_DIR"
fi

# ── Preserve the production ConvTran result ──────────────────────────────────
if [[ -d "$PROD_CONVTRAN" ]]; then
    echo "  Backing up production ConvTran result -> $(basename "$PROD_BACKUP")"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        # Don't let a stale-backup cleanup failure abort the run under `set -e`;
        # verify the backup actually succeeded before touching anything else.
        rm -rf "$PROD_BACKUP" 2>/dev/null || true
        if [[ -e "$PROD_BACKUP" ]]; then
            echo "ERROR: a stale backup exists at $PROD_BACKUP and could not be removed." >&2
            echo "Remove it manually, then re-run. Refusing to proceed so the current" >&2
            echo "production ConvTran result cannot be lost." >&2
            exit 1
        fi
        cp -r "$PROD_CONVTRAN" "$PROD_BACKUP"
        if [[ ! -f "$PROD_BACKUP/convtran_results.csv" ]]; then
            echo "ERROR: backup of $PROD_CONVTRAN failed (no convtran_results.csv copied)." >&2
            echo "Refusing to proceed — the sweep would overwrite an unbacked-up result." >&2
            exit 1
        fi
    fi
fi

restore_production() {
    [[ -d "$PROD_BACKUP" ]] || return 0
    echo ""
    echo "  Restoring production ConvTran result to convtran_prauc/"
    # Clear the destination first, but never leave it half-deleted: if the
    # removal fails (read-only mount, permissions, open file handle), bail out
    # with the backup intact rather than proceeding into a failed mv that would
    # lose the production result entirely.
    rm -rf "$PROD_CONVTRAN" 2>/dev/null || true
    if [[ -e "$PROD_CONVTRAN" ]]; then
        echo "  WARNING: could not clear $PROD_CONVTRAN." >&2
        echo "  Your production result is UNCHANGED and the backup is preserved at:" >&2
        echo "    $PROD_BACKUP" >&2
        echo "  Restore it manually if convtran_prauc/ holds a sweep result." >&2
        return 0
    fi
    mv "$PROD_BACKUP" "$PROD_CONVTRAN"
}
# Restore even if the sweep is interrupted (Ctrl-C) or a run fails.
if [[ "$DRY_RUN" -eq 0 ]]; then
    trap restore_production EXIT INT TERM
fi

# ── Sweep ────────────────────────────────────────────────────────────────────
for BS in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  ConvTran batch_size=$BS"
    echo "──────────────────────────────────────────────────────────"

    DEST="$SWEEP_DIR/convtran_bs${BS}"
    if [[ -d "$DEST" && -f "$DEST/convtran_results.csv" ]]; then
        echo "  [skip] $DEST already has results (delete it to re-run)"
        continue
    fi

    PIPELINE_ARGS=(--from-phase "$FROM_PHASE" --run-name "$RUN_NAME" --convtran-batch-size "$BS")
    [[ -n "$FPS" ]]          && PIPELINE_ARGS+=(--fps "$FPS")
    [[ -n "$FEATURES_CSV" ]] && PIPELINE_ARGS+=(--features-csv "$FEATURES_CSV")
    [[ "$TEST_RUN" -eq 1 ]]  && PIPELINE_ARGS+=(--test-run)

    # Invalidate only the steps whose output depends on the ConvTran batch size.
    # Checkpoints do not encode batch size, so without this the pipeline skips
    # ConvTran and silently reuses the previous run's numbers.
    echo "  Invalidating checkpoints: p3_convtran, p3_compare, p3_pr_curves"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] rm -f $CKPTS/{p3_convtran,p3_compare,p3_pr_curves}.done"
        echo "  [dry-run] bash run_pipeline_review.sh ${PIPELINE_ARGS[*]}"
        echo "  [dry-run] mv $PROD_CONVTRAN -> $DEST"
        continue
    fi

    rm -f "$CKPTS/p3_convtran.done" "$CKPTS/p3_compare.done" "$CKPTS/p3_pr_curves.done"

    bash run_pipeline_review.sh "${PIPELINE_ARGS[@]}"

    if [[ ! -f "$PROD_CONVTRAN/convtran_results.csv" ]]; then
        echo "  WARNING: no convtran_results.csv produced for batch_size=$BS — skipping archive." >&2
        continue
    fi
    rm -rf "$DEST"
    mv "$PROD_CONVTRAN" "$DEST"
    echo "  Archived -> $DEST"
done

# ── Aggregate ────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "[dry-run] would aggregate $SWEEP_DIR/convtran_bs*/convtran_results.csv"
    exit 0
fi

echo ""
echo "──────────────────────────────────────────────────────────"
echo "  Aggregating sweep results"
echo "──────────────────────────────────────────────────────────"
SWEEP_DIR="$SWEEP_DIR" "$PYTHON" - <<'PYEOF'
import os, re
from pathlib import Path
import pandas as pd

sweep = Path(os.environ["SWEEP_DIR"])
rows = []
for d in sorted(sweep.glob("convtran_bs*")):
    csv = d / "convtran_results.csv"
    if not csv.exists():
        continue
    r = pd.read_csv(csv).iloc[0]
    m = re.search(r"convtran_bs(\d+)", d.name)
    bs = int(r["batch_size"]) if "batch_size" in r.index and pd.notna(r.get("batch_size")) \
         else (int(m.group(1)) if m else None)
    rows.append({
        "batch_size": bs,
        "n_params": int(r["n_params"]),
        "selected_epochs": int(r["selected_epochs"]),
        "threshold": float(r["threshold"]),
        "precision": float(r["precision"]),
        "recall": float(r["recall"]),
        "f1": float(r["f1"]),
        "pr_auc": float(r["pr_auc"]),
        "pr_auc_lo": float(r["pr_auc_lo"]),
        "pr_auc_hi": float(r["pr_auc_hi"]),
        "latency_mean_ms": float(r["latency_mean_ms"]),
        "device": str(r.get("device", "")),
    })

if not rows:
    print("  No sweep results found yet.")
    raise SystemExit(0)

df = pd.DataFrame(rows).sort_values("batch_size").reset_index(drop=True)
out = sweep / "convtran_batch_size_summary.csv"
df.to_csv(out, index=False)

print(f"\n  {'batch':>6} {'ep':>4} {'thr':>6}  {'PR-AUC':>7}  {'95% CI':^17}  "
      f"{'F1':>7} {'Rec':>7} {'Prec':>7} {'ms/samp':>9}")
print("  " + "─" * 82)
for _, r in df.iterrows():
    ci = f"[{r['pr_auc_lo']:.3f}–{r['pr_auc_hi']:.3f}]"
    print(f"  {int(r['batch_size']):>6} {int(r['selected_epochs']):>4} {r['threshold']:>6.3f}  "
          f"{r['pr_auc']:>7.4f}  {ci:^17}  {r['f1']:>7.4f} {r['recall']:>7.4f} "
          f"{r['precision']:>7.4f} {r['latency_mean_ms']:>9.3f}")

print(f"\n  PR-AUC range across batch sizes: {df['pr_auc'].max() - df['pr_auc'].min():.4f}")
print(f"  (production default 128; official ConvTran default 16)")
print(f"\n  Summary -> {out}")
PYEOF

echo ""
echo "  Sweep complete. Production convtran_prauc/ restored on exit."
