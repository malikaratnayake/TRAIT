#!/usr/bin/env bash
# =============================================================================
# clean.sh — Delete one pipeline run folder so that run can start from scratch
# =============================================================================
#
# Deletes data/results/<RUN_NAME>/ (features, models, figures, checkpoints and
# the cached --fps). Nothing else in data/ is touched: the dataset, the split
# files and every other run folder are kept. Run folders containing read-only
# files, such as the locked results reported in the paper, are refused.
#
# Usage:
#   bash clean.sh RUN_NAME              # dry run — shows what would be deleted
#   bash clean.sh RUN_NAME --confirm    # delete data/results/RUN_NAME/
#   bash clean.sh --logs [--confirm]    # delete the logs/ folder
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_NAME=""
CLEAN_LOGS=0
CONFIRM=0
for arg in "$@"; do
    case "$arg" in
        --confirm) CONFIRM=1 ;;
        --logs)    CLEAN_LOGS=1 ;;
        -h|--help)
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 && !/^#/ { exit }' "$0"
            exit 0 ;;
        -*)        echo "Unknown option: $arg" >&2; exit 1 ;;
        *)         RUN_NAME="$arg" ;;
    esac
done

if [[ -z "$RUN_NAME" && "$CLEAN_LOGS" -eq 0 ]]; then
    echo "Usage: bash clean.sh RUN_NAME [--confirm]   or   bash clean.sh --logs [--confirm]" >&2
    exit 1
fi

TARGETS=()
if [[ -n "$RUN_NAME" ]]; then
    if [[ "$RUN_NAME" == */* || "$RUN_NAME" == "." || "$RUN_NAME" == ".." ]]; then
        echo "RUN_NAME must be the name of a folder in data/results/, not a path: $RUN_NAME" >&2
        exit 1
    fi
    RUN_DIR="data/results/$RUN_NAME"
    if [[ ! -d "$RUN_DIR" ]]; then
        echo "No such run folder: $RUN_DIR" >&2
        exit 1
    fi
    if [[ -n "$(find "$RUN_DIR" ! -perm -u+w -print -quit)" ]]; then
        echo "Refusing to delete $RUN_DIR: it contains read-only (locked) files." >&2
        exit 1
    fi
    TARGETS+=("$RUN_DIR")
fi
if [[ "$CLEAN_LOGS" -eq 1 && -d logs ]]; then
    TARGETS+=("logs")
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "Nothing to delete."
    exit 0
fi

if [[ "$CONFIRM" -eq 0 ]]; then
    echo "Dry run — nothing will be deleted. Add --confirm to delete:"
    for t in "${TARGETS[@]}"; do
        echo "  $t"
    done
    exit 0
fi

for t in "${TARGETS[@]}"; do
    rm -rf "$t"
    echo "  removed  $t"
done
