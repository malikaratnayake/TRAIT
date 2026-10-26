"""
Class-conditional trajectory overlays (side by side, shared y axis).

Left panel  : Non-insect trajectories (orange)
Right panel : Insect trajectories (blue)

All tracks translated to start at origin (0, 0).
Path colour fades from light (early frames) → dark (late frames).
End positions shown as a scatter cloud.

Usage
-----
Run from the TRAIT project root:

    python src/figures/plot_class_trajectory_overlays.py

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
DEFAULT_OUT = ROOT / "data" / "results"

BLUE   = "#2166AC"
ORANGE = "#D6604D"
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = AXIS_LABEL_SIZE


parser = argparse.ArgumentParser(description="Plot class-conditional trajectory overlays.")
parser.add_argument("--tracks-dir", default=str(DEFAULT_TRACKS),
                    help="Directory containing per-track CSVs.")
parser.add_argument("--splits-csv", default=str(DEFAULT_SPLITS),
                    help="Splits CSV containing filename and label columns.")
parser.add_argument("--output-dir", default=str(DEFAULT_OUT),
                    help="Directory where figure files will be written.")
args = parser.parse_args()

TRACKS = Path(args.tracks_dir).expanduser().resolve()
SPLITS = Path(args.splits_csv).expanduser().resolve()
OUT = Path(args.output_dir).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)

if not TRACKS.exists():
    raise SystemExit(f"Tracks directory not found: {TRACKS}")
if not SPLITS.exists():
    raise SystemExit(f"Splits CSV not found: {SPLITS}")


# ── data helpers ──────────────────────────────────────────────────────────────
def load_track(fname: str):
    """Load a track CSV; return DataFrame or None if too short."""
    path = TRACKS / fname
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


def collect_tracks(fns: list, n_want: int) -> list:
    """Load and align up to *n_want* tracks from *fns*."""
    tracks = []
    for fn in fns:
        df = load_track(fn)
        if df is None:
            continue
        tracks.append(align_track(df))
        if len(tracks) == n_want:
            break
    return tracks


# ── load splits and collect tracks ────────────────────────────────────────────
splits   = pd.read_csv(SPLITS)
all_lbl  = splits[["filename", "label"]].copy()

np.random.seed(0)
N_OVERLAY = 200

ins_fns = all_lbl[all_lbl["label"] == "insect"]["filename"].tolist()
ni_fns  = all_lbl[all_lbl["label"] == "not_insect"]["filename"].tolist()
np.random.shuffle(ins_fns)
np.random.shuffle(ni_fns)

ins_tracks = collect_tracks(ins_fns, N_OVERLAY)
ni_tracks  = collect_tracks(ni_fns,  N_OVERLAY)
print(f"Loaded {len(ins_tracks)} insect, {len(ni_tracks)} non-insect tracks")

# ── shared axis limits (92nd-percentile of combined absolute extent) ──────────
all_coords = np.abs(
    np.concatenate(
        [t[0] for t in ins_tracks + ni_tracks]
        + [t[1] for t in ins_tracks + ni_tracks]
    )
)
lim = float(np.percentile(all_coords, 92))


# ── panel drawing function ────────────────────────────────────────────────────
def draw_panel(ax, tracks: list, base_col: str) -> None:
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


# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_ni, ax_ins) = plt.subplots(
    1, 2, figsize=(12, 6.5), sharex=True, sharey=True
)
fig.subplots_adjust(wspace=0.05, bottom=0.2)

draw_panel(ax_ni,  ni_tracks,  ORANGE)
draw_panel(ax_ins, ins_tracks, BLUE)

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
    out_path = OUT / f"class_trajectory_overlays.{ext}"
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    print(f"  Saved → {out_path}")

plt.close(fig)
print("Done.")
