#!/usr/bin/env bash
# =============================================================================
# clean.sh — Remove all generated outputs so the pipeline runs from scratch
# =============================================================================
#
# Deletes:
#   data/results/          — all run folders (features, models, figures, ckpts)
#   data/splits_folds.csv  — Phase 2 splits
#   data/pipeline.cfg      — cached fps setting
#   logs/                  — pipeline log files
#
# Preserves:
#   data/tracks/                  — raw track CSVs (never regenerated)
#   data/track_classification.csv — labels file (never regenerated)
#
# Usage:
#   bash clean.sh            # dry run — shows what would be deleted
#   bash clean.sh --confirm  # actually delete
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIRM=0
[[ "${1:-}" == "--confirm" ]] && CONFIRM=1

# Targets to remove
TARGETS=(
    "data/results"
    "data/splits_folds.csv"
    "data/pipeline.cfg"
    "logs"
)

if [[ "$CONFIRM" -eq 0 ]]; then
    echo "Dry run — nothing will be deleted. Pass --confirm to proceed."
    echo ""
    echo "Would remove:"
    for t in "${TARGETS[@]}"; do
        [[ -e "$t" ]] && echo "  $t" || echo "  $t  (not found — skip)"
    done
    echo ""
    echo "Would preserve:"
    echo "  data/tracks/                   (raw track CSVs)"
    echo "  data/track_classification.csv  (labels)"
    echo ""
    echo "Run:  bash clean.sh --confirm"
    exit 0
fi

echo "Cleaning generated outputs …"
for t in "${TARGETS[@]}"; do
    if [[ -e "$t" ]]; then
        rm -rf "$t"
        echo "  removed  $t"
    else
        echo "  skipped  $t  (not found)"
    fi
done
echo ""
echo "Done. Run 'bash run_pipeline.sh --fps <FPS>' to start fresh."
