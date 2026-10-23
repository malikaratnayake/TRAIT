#!/bin/bash
# Pre-flight dependency check — run BEFORE submitting the 48h job.
# The pipeline degrades gracefully on missing deps (warns, marks step done,
# continues), so a missing package = silently absent results after 48 hours.
PY="${PYTHON:-python3}"
echo "Interpreter: $($PY -c 'import sys; print(sys.executable)')"
echo "Python:      $($PY -c 'import sys; print(sys.version.split()[0])')"
echo ""
fail=0
check() {  # $1=import name  $2=what breaks if missing
  if $PY -c "import $1" >/dev/null 2>&1; then
    v=$($PY -c "import $1; print(getattr($1,'__version__','?'))" 2>/dev/null)
    printf "  OK    %-14s %-10s\n" "$1" "$v"
  else
    printf "  MISS  %-14s %-10s -> %s\n" "$1" "" "$2"; fail=1
  fi
}
check numpy       "everything"
check scipy       "SG smoothing / feature extraction"
check pandas      "everything"
check sklearn     "all models, metrics, bootstrap CIs"
check xgboost     "XGBoost row in Table 3 + ALL of Phase 4 + ALL of Phase 5"
check matplotlib  "PR-curve and ablation figures"
check torch       "1D-CNN + Transformer x2 + ConvTran (4 of 9 models)"
check aeon        "miniROCKET row in Table 3"
check numba       "aeon/miniROCKET backend"
echo ""
[ $fail -eq 0 ] && echo "All dependencies present — safe to submit." \
               || { echo "MISSING DEPS ABOVE: the pipeline will SKIP those steps"; \
                    echo "silently (warn + mark done) and you will lose those results."; exit 1; }
