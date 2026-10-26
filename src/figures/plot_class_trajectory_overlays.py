"""
src/figures/plot_class_trajectory_overlays.py
=============================================
Phase 5 — Class-conditional trajectory overlays (Figure 3).

Draws 200 randomly sampled tracks per class (seed 0) side by side: non-insect
tracks on the left in orange, insect tracks on the right in blue. Every track
is smoothed and translated so it starts at the origin; its colour fades from
light (early frames) to dark (late frames), and end positions are shown as
dots. Both panels share one axis range (the 92nd percentile of all absolute
coordinates), with y increasing downwards as in the camera image.

Smoothing uses src/figures/_track_utils.sg_smooth, which leaves the first and
last few points of a track unsmoothed; this reproduces the published figure.

Usage
-----
Run from the TRAIT project root:

    python src/figures/plot_class_trajectory_overlays.py \\
        --splits-csv data/splits_folds.csv \\
        --output-dir data/results/trait_results/figures

Outputs
-------
    <output-dir>/class_trajectory_overlays.pdf
    <output-dir>/class_trajectory_overlays.png
"""

import argparse
import warnings
from pathlib import Path

try:
    from ._matplotlib_config import configure_matplotlib_cache
    from ._track_utils import sg_smooth
except ImportError:
    from _matplotlib_config import configure_matplotlib_cache
    from _track_utils import sg_smooth

configure_matplotlib_cache()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── paths (relative to project root) ─────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]   # …/TRAIT
DEFAULT_TRACKS = ROOT / "data" / "tracks"
DEFAULT_SPLITS = ROOT / "data" / "splits_folds.csv"
DEFAULT_OUT = ROOT / "data" / "results" / "trait_results" / "figures"

BLUE   = "#2166AC"
ORANGE = "#D6604D"
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = AXIS_LABEL_SIZE
N_OVERLAY = 200


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Plot class-conditional trajectory overlays (Figure 3).")
    parser.add_argument("--tracks-dir", default=str(DEFAULT_TRACKS),
                        help="Directory containing per-track CSVs.")
    parser.add_argument("--splits-csv", default=str(DEFAULT_SPLITS),
                        help="Splits CSV containing filename and label columns.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT),
                        help="Directory where figure files will be written.")
    return parser.parse_args()


# ── data helpers ──────────────────────────────────────────────────────────────
def load_track(tracks_dir: Path, fname: str):
    """Load a track CSV; return DataFrame or None if too short."""
    path = tracks_dir / fname
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) >= 10 else None
    except Exception:
        return None


def align_track(df: pd.DataFrame):
    """Smooth + translate so the track starts at (0, 0)."""
    sx = sg_smooth(df["x"].values.astype(float))
    sy = sg_smooth(df["y"].values.astype(float))
    sx -= sx[0]
    sy -= sy[0]
    return sx, sy


def collect_tracks(tracks_dir: Path, fns: list, n_want: int) -> list:
    """Load and align up to *n_want* tracks from *fns*."""
    tracks = []
    for fn in fns:
        df = load_track(tracks_dir, fn)
        if df is None:
            continue
        tracks.append(align_track(df))
        if len(tracks) == n_want:
            break
    return tracks


# ── panel drawing function ────────────────────────────────────────────────────
def draw_panel(ax, tracks: list, base_col: str, lim: float) -> None:
    """Draw one class's tracks, end points and origin marker on ax, within ±lim pixels."""
    cmap  = mcolors.LinearSegmentedColormap.from_list(
        "tc", ["#EEEEEE", base_col]
    )
    alpha = 0.40

    for sx, sy in tracks:
        t = np.linspace(0, 1, len(sx))
        for i in range(len(sx) - 1):
            ax.plot(
                sx[i:i+2], sy[i:i+2],
                color=cmap(0.25 + 0.75 * t[i]),
                linewidth=0.65, alpha=alpha, zorder=2,
            )

    # end-position scatter
    ends_x = [sx[-1] for sx, _ in tracks]
    ends_y = [sy[-1] for _, sy in tracks]
    ax.scatter(ends_x, ends_y, s=9, color=base_col,
               alpha=0.30, zorder=3)

    # origin marker + crosshairs
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.25, zorder=1)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.25, zorder=1)
    ax.plot(0, 0, "o", ms=7, color="black", zorder=6)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.invert_yaxis()   # match camera convention: y increases downward
    ax.set_aspect("equal")
    ax.set_xlabel("x (pixels)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def main() -> None:
    """Sample tracks per class and save the overlay figure as PDF and PNG."""
    args = parse_args()
    tracks_dir = Path(args.tracks_dir).expanduser().resolve()
    splits_csv = Path(args.splits_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not tracks_dir.exists():
        raise SystemExit(f"Tracks directory not found: {tracks_dir}")
    if not splits_csv.exists():
        raise SystemExit(f"Splits CSV not found: {splits_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load splits and collect tracks ────────────────────────────────────────
    splits   = pd.read_csv(splits_csv)
    all_lbl  = splits[["filename", "label"]].copy()

    np.random.seed(0)
    ins_fns = all_lbl[all_lbl["label"] == "insect"]["filename"].tolist()
    ni_fns  = all_lbl[all_lbl["label"] == "not_insect"]["filename"].tolist()
    np.random.shuffle(ins_fns)
    np.random.shuffle(ni_fns)

    ins_tracks = collect_tracks(tracks_dir, ins_fns, N_OVERLAY)
    ni_tracks  = collect_tracks(tracks_dir, ni_fns,  N_OVERLAY)
    print(f"Loaded {len(ins_tracks)} insect, {len(ni_tracks)} non-insect tracks")

    # ── shared axis limits (92nd-percentile of combined absolute extent) ──────
    all_coords = np.abs(
        np.concatenate(
            [t[0] for t in ins_tracks + ni_tracks]
            + [t[1] for t in ins_tracks + ni_tracks]
        )
    )
    lim = float(np.percentile(all_coords, 92))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax_ni, ax_ins) = plt.subplots(
        1, 2, figsize=(12, 6.5), sharex=True, sharey=True
    )
    fig.subplots_adjust(wspace=0.05, bottom=0.2)

    draw_panel(ax_ni,  ni_tracks,  ORANGE, lim)
    draw_panel(ax_ins, ins_tracks, BLUE, lim)

    ax_ni.set_ylabel("y (pixels)", fontsize=AXIS_LABEL_SIZE)   # shared y title
    ax_ins.tick_params(axis="y", labelleft=False)                 # shared y ticks

    legend_handles = [
        plt.Line2D([0], [0], color=ORANGE, linewidth=2.2, label="Non-insect track"),
        plt.Line2D([0], [0], color=BLUE, linewidth=2.2, label="Insect track"),
        plt.Line2D([0], [0], marker="o", color="black", ms=6,
                   linestyle="none", label="Start (all tracks)"),
        plt.Line2D([0], [0], marker="o", color="0.35", ms=6,
                   linestyle="none", alpha=0.7, label="End"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.04),
    )

    for ext in ("pdf", "png"):
        out_path = out_dir / f"class_trajectory_overlays.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=180)
        print(f"  Saved → {out_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
