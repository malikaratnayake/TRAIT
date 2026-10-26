"""
src/convtran_batch_size_study.py
================================
Supplementary study — ConvTran batch-size sensitivity (Supp. Table 11).

src/train_convtran.py uses ConvTran's published architecture defaults but is
trained with batch size 128 and AdamW, the same training budget as the
1D-CNN, rather than the published recipe (batch size 16, RAdam), which was
tuned on much smaller UCR/UEA benchmark datasets. This script retrains
ConvTran at several batch sizes to check that the reported result does not
depend on that choice.

Each batch size runs src/train_convtran.py as a separate process, writing to
<output-dir>/convtran_batch_size_study/bs_<N>/ (with its own train.log), and
the results are collected into one summary table. The reported ConvTran
result in convtran_prauc/ is not changed. Batch sizes that already have
results are skipped unless --overwrite is given, and a failed batch size does
not stop the others. The pipeline runs this study when given
--convtran-batch-sweep; it can also be run on its own.

Each batch size is a full training run, and smaller batches take longer
(batch 16 is roughly 8x slower per epoch than batch 128).

Output
------
  <output-dir>/convtran_batch_size_study/convtran_batch_size_summary.csv

Usage
-----
  python src/convtran_batch_size_study.py \\
      --batch-sizes 16 32 64 96 128 \\
      --output-dir data/results/trait_results

  Quick check that it runs (not for reporting):
  python src/convtran_batch_size_study.py --batch-sizes 16 32 --limit 300 --epochs 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_SIZES = [16, 32, 64, 96, 128]


def run_one_batch_size(python_exe: str, batch_size: int, study_dir: Path, args) -> Path:
    """Invoke src/train_convtran.py as a subprocess for one batch size.

    Returns the path to that run's results CSV. Raises CalledProcessError on
    failure (caught by the caller so one OOM/failed batch size doesn't abort
    the whole sweep).
    """
    model_name = f"ConvTran_bs{batch_size}_PR_AUC"
    run_dir = study_dir / f"bs_{batch_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "convtran_results.csv"

    if results_csv.exists() and not args.overwrite:
        print(f"  [skip] batch_size={batch_size} — {results_csv.name} already exists "
              f"(use --overwrite to re-run)")
        return results_csv

    cmd = [
        python_exe, "src/train_convtran.py",
        "--max_len", str(args.max_len),
        "--epochs", str(args.epochs),
        "--batch-size", str(batch_size),
        "--early-stop-metric", args.early_stop_metric,
        "--model-name", model_name,
        "--output-dir", str(run_dir),
        "--device", args.device,
    ]
    for flag, value in (("--tracks-dir", args.tracks_dir),
                        ("--splits-csv", args.splits_csv),
                        ("--labels-csv", args.labels_csv)):
        if value is not None:
            cmd += [flag, str(Path(value).expanduser().resolve())]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.deterministic:
        cmd += ["--deterministic"]

    log_path = run_dir / "train.log"
    try:
        log_display = log_path.relative_to(PROJECT_ROOT)
    except ValueError:
        log_display = log_path  # output-dir is outside the repo — show the absolute path instead
    print(f"  batch_size={batch_size:<4} -> {run_dir}  (log: {log_display})")
    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"    done in {elapsed/60:.1f} min")
    return results_csv


def run_sweep(args) -> pd.DataFrame:
    """Train ConvTran at each batch size and write convtran_batch_size_summary.csv."""
    study_dir = Path(args.output_dir).expanduser().resolve() / "convtran_batch_size_study"
    study_dir.mkdir(parents=True, exist_ok=True)
    python_exe = args.python or sys.executable

    print("\n── ConvTran batch-size sensitivity (Supp. Table 11) ──────────")
    print(f"  Batch sizes : {args.batch_sizes}")
    print(f"  max_len={args.max_len}  epochs(budget)={args.epochs}  device={args.device}")
    print(f"  Output      : {study_dir}\n")

    rows = []
    for bs in args.batch_sizes:
        try:
            results_csv = run_one_batch_size(python_exe, bs, study_dir, args)
        except subprocess.CalledProcessError as exc:
            print(f"  [FAILED] batch_size={bs} — subprocess exited {exc.returncode}. "
                  f"See {study_dir / f'bs_{bs}' / 'train.log'}. Skipping this batch size.")
            rows.append({"batch_size": bs, "status": "failed"})
            continue

        r = pd.read_csv(results_csv).iloc[0]
        rows.append({
            "batch_size": bs,
            "status": "ok",
            "n_params": int(r["n_params"]),
            "selected_epochs": int(r["selected_epochs"]),
            "threshold": float(r["threshold"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1"]),
            "f1_lo": float(r["f1_lo"]),
            "f1_hi": float(r["f1_hi"]),
            "pr_auc": float(r["pr_auc"]),
            "pr_auc_lo": float(r["pr_auc_lo"]),
            "pr_auc_hi": float(r["pr_auc_hi"]),
            "latency_mean_ms": float(r["latency_mean_ms"]),
        })

    summary = pd.DataFrame(rows)
    summary_path = study_dir / "convtran_batch_size_summary.csv"
    summary.to_csv(summary_path, index=False)

    ok = summary[summary["status"] == "ok"]
    if not ok.empty:
        print(f"\n  {'batch':>6} {'params':>8} {'ep':>4} {'thr':>6}  {'PR-AUC':>7}  "
              f"{'95% CI':^16}  {'F1':>7}  {'ms/samp':>8}")
        print("  " + "─" * 76)
        for _, r in ok.iterrows():
            ci = f"[{r['pr_auc_lo']:.3f}–{r['pr_auc_hi']:.3f}]"
            print(f"  {int(r['batch_size']):>6} {int(r['n_params']):>8,} {int(r['selected_epochs']):>4} "
                  f"{r['threshold']:>6.3f}  {r['pr_auc']:>7.4f}  {ci:^16}  {r['f1']:>7.4f}  "
                  f"{r['latency_mean_ms']:>8.3f}")

        prauc_range = ok["pr_auc"].max() - ok["pr_auc"].min()
        print(f"\n  PR-AUC range across batch sizes tested: {prauc_range:.4f}")
        print("  (reported ConvTran uses batch_size=128; published ConvTran recipe uses 16)")

    failed = summary[summary["status"] == "failed"]
    if not failed.empty:
        print(f"\n  Failed batch sizes: {list(failed['batch_size'])} — see individual train.log files")

    print(f"\n  Summary -> {summary_path}")
    return summary


def parse_args():
    """Parse command-line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
                   help=f"Batch sizes to sweep (default: {DEFAULT_BATCH_SIZES})")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60, help="Epoch budget per run (early stopping still applies)")
    p.add_argument("--early-stop-metric", choices=["f1", "pr_auc"], default="pr_auc")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu",
                   help="Default cpu — latency must stay comparable to the main ConvTran result")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Enabled by default so batch-size is the only thing varying run-to-run")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--limit", type=int, default=None, help="Smoke-testing only — caps tracks per split")
    p.add_argument("--output-dir", default="data/results/trait_results")
    p.add_argument("--tracks-dir", default=None, help="Passed to train_convtran.py (default: its own)")
    p.add_argument("--splits-csv", default=None, help="Passed to train_convtran.py (default: its own)")
    p.add_argument("--labels-csv", default=None, help="Passed to train_convtran.py (default: its own)")
    p.add_argument("--python", default=None, help="Python interpreter to use (default: current interpreter)")
    p.add_argument("--overwrite", action="store_true", help="Re-run batch sizes that already have results")
    return p.parse_args()


def main():
    """Run the batch-size sweep."""
    args = parse_args()
    run_sweep(args)
    print("\n✓ ConvTran batch-size sensitivity study complete")


if __name__ == "__main__":
    main()
