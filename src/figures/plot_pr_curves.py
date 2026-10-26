"""
Plot Phase 3 precision-recall curves from saved curve CSV files.

Inputs are the per-model `*_pr_curve.csv` files written by:
  - src/models/majority_baseline.py
  - src/train_rocket.py
  - src/train_fcn.py
  - src/train_convtran.py

Outputs:
  - phase3_precision_recall_curves.csv
  - phase3_precision_recall_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._matplotlib_config import configure_matplotlib_cache
except ImportError:
    from _matplotlib_config import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULTS = Path("data/results/six_domain_v1")
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 11


# PLOT_ORDER is the actual filter: any model whose name is not listed there
# is silently dropped from both the figure and the combined CSV, regardless
# of whether it has a DISPLAY_NAMES entry. Keep the two in sync.
DISPLAY_NAMES = {
    "majority": "Majority baseline",
    "logistic_regression_prauc": "Logistic Regression",
    "random_forest_prauc": "Random Forest",
    "xgboost_prauc": "XGBoost",
    "FCN_PR_AUC": "1D-CNN",
    "MiniROCKET_PR_AUC": "MiniROCKET",
    "ConvTran_PR_AUC": "ConvTran",
}

PLOT_ORDER = [
    "majority", "logistic_regression_prauc", "random_forest_prauc",
    "xgboost_prauc", "FCN_PR_AUC", "MiniROCKET_PR_AUC", "ConvTran_PR_AUC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 3 precision-recall curves")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def load_curves(results_dir: Path) -> pd.DataFrame:
    files = sorted(
        p for p in results_dir.rglob("*_pr_curve.csv")
        if p.name != "phase2_precision_recall_curves.csv"
    )
    if not files:
        raise SystemExit(f"No *_pr_curve.csv files found under {results_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        required = {"model", "precision", "recall"}
        missing = required - set(df.columns)
        if missing:
            print(f"Skipping {path}: missing {sorted(missing)}")
            continue
        df["source_file"] = str(path.relative_to(results_dir))
        frames.append(df)

    if not frames:
        raise SystemExit(f"No valid precision-recall curve CSVs found under {results_dir}")
    curves = pd.concat(frames, ignore_index=True)
    curves = curves[curves["model"].isin(PLOT_ORDER)].copy()
    if curves.empty:
        raise SystemExit(
            "No requested Phase 2 model curves found. Expected: "
            + ", ".join(PLOT_ORDER)
        )
    curves["model"] = pd.Categorical(curves["model"], categories=PLOT_ORDER, ordered=True)
    return curves.sort_values(["model", "recall"]).reset_index(drop=True)


def plot_curves(curves: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)

    for model in PLOT_ORDER:
        df = curves[curves["model"] == model].sort_values("recall")
        if df.empty:
            continue
        label = DISPLAY_NAMES.get(str(model), str(model))
        ap = df["average_precision"].max() if "average_precision" in df else None
        if pd.notna(ap):
            label = f"{label} (PR-AUC={ap:.3f})"
        ax.plot(df["recall"], df["precision"], linewidth=1.8, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Precision", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower left", fontsize=LEGEND_SIZE, frameon=False)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else \
        results_dir / "phase2_precision_recall_curves.csv"
    output_png = Path(args.output_png).expanduser().resolve() if args.output_png else \
        results_dir / "phase2_precision_recall_curve.png"

    curves = load_curves(results_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    curves_for_csv = curves.copy()
    curves_for_csv["model"] = curves_for_csv["model"].astype(str).replace(DISPLAY_NAMES)
    curves_for_csv.to_csv(output_csv, index=False)
    plot_curves(curves, output_png)

    print(f"Combined PR curve data -> {output_csv}")
    print(f"PR curve figure        -> {output_png}")


if __name__ == "__main__":
    main()
