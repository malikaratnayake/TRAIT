"""
src/figures/plot_session_trajectories.py
========================================
Phase 5 — All trajectories from one recording (Figure 2).

Ranks recordings (station + start time, taken from the track filenames) by
their number of tracks, keeps those with at least --min-insects insect
tracks, and plots every track of the recording at --rank (default 1: the
busiest such recording; in the paper, 8 insect and 262 non-insect tracks
from Farm A). Insect tracks are drawn in blue on top of non-insect tracks in
red, on the full camera frame (--frame-size, default 1920 × 1080), with the
y axis flipped so the image appears upright.

Smoothing uses src/figures/_track_utils.sg_smooth, which leaves the first and
last few points of a track unsmoothed; this reproduces the published figure.

Usage
-----
Run from the TRAIT project root:

    # Rank-1 recording (most tracks, ≥6 insects)
    python src/figures/plot_session_trajectories.py

    # Rank-2 recording
    python src/figures/plot_session_trajectories.py --rank 2

    # Custom minimum insect count
    python src/figures/plot_session_trajectories.py --min-insects 10 --rank 1

Outputs
-------
    <output-dir>/session_trajectories_rank<N>.pdf
    <output-dir>/session_trajectories_rank<N>.png
"""

import argparse
import re
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

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]   # …/TRAIT
DEFAULT_TRACKS = ROOT / "data" / "tracks"
DEFAULT_SPLITS = ROOT / "data" / "splits_folds.csv"
DEFAULT_OUT = ROOT / "data" / "results" / "trait_results" / "figures"

BLUE = "#2166AC"
RED  = "#B2182B"

AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = AXIS_LABEL_SIZE


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Plot all trajectories from one recording (Figure 2).")
    parser.add_argument("--rank",        type=int, default=1,
                        help="Rank of session to plot (1 = most tracks). Default: 1.")
    parser.add_argument("--min-insects", type=int, default=6,
                        help="Minimum number of insect tracks required. Default: 6.")
    parser.add_argument("--frame-size", type=int, nargs=2, default=[1920, 1080],
                        metavar=("WIDTH", "HEIGHT"),
                        help="Camera frame size in pixels. Default: 1920 1080.")
    parser.add_argument("--tracks-dir", default=str(DEFAULT_TRACKS),
                        help="Directory containing per-track CSVs.")
    parser.add_argument("--splits-csv", default=str(DEFAULT_SPLITS),
                        help="Splits CSV containing filename and label columns.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT),
                        help="Directory where figure files will be written.")
    return parser.parse_args()


# ── session selection ─────────────────────────────────────────────────────────
def extract_session(fn: str) -> str:
    """Return 'StationKey_YYMMDD_HHMMSS' from a track filename."""
    m = re.match(r"(.+?)_(\d{6}_\d{6})_", fn)
    return f"{m.group(1)}_{m.group(2)}" if m else fn


def load_smooth(tracks_dir: Path, fn: str, frame_height: int):
    """Return (sx, sy) arrays, with y flipped to point up, or (None, None) on failure."""
    path = tracks_dir / fn
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path)
        if len(df) < 5:
            return None, None
        return (
            sg_smooth(df["x"].values.astype(float)),
            frame_height - sg_smooth(df["y"].values.astype(float)),
        )
    except Exception:
        return None, None


def main() -> None:
    """Pick the recording, then save its trajectory figure as PDF and PNG."""
    args = parse_args()
    rank        = args.rank
    min_insects = args.min_insects
    frame_width, frame_height = args.frame_size
    tracks_dir = Path(args.tracks_dir).expanduser().resolve()
    splits_csv = Path(args.splits_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not tracks_dir.exists():
        raise SystemExit(f"Tracks directory not found: {tracks_dir}")
    if not splits_csv.exists():
        raise SystemExit(f"Splits CSV not found: {splits_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(splits_csv)
    splits["session"] = splits["filename"].apply(extract_session)

    session_stats = (
        splits.groupby("session")
        .agg(
            total    = ("filename", "count"),
            n_insect = ("label", lambda x: (x == "insect").sum()),
            n_nonins = ("label", lambda x: (x == "not_insect").sum()),
        )
        .reset_index()
    )

    ranked = (
        session_stats[session_stats["n_insect"] >= min_insects]
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )

    if rank > len(ranked):
        raise ValueError(
            f"Only {len(ranked)} sessions have ≥{min_insects} insects; "
            f"cannot select rank {rank}."
        )

    best       = ranked.iloc[rank - 1]
    session_key = best["session"]
    n_ins      = int(best["n_insect"])
    n_ni       = int(best["n_nonins"])
    n_total    = int(best["total"])
    prevalence = 100 * n_ins / n_total

    print(f"Rank {rank} session (≥{min_insects} insects): {session_key}")
    print(f"  Total tracks   : {n_total}  ({n_ins} insect, {n_ni} non-insect)")
    print(f"  Prevalence     : {prevalence:.1f}%")

    # ── load and smooth tracks ────────────────────────────────────────────────
    session_rows = splits[splits["session"] == session_key]
    ins_fns = session_rows[session_rows["label"] == "insect"]["filename"].tolist()
    ni_fns  = session_rows[session_rows["label"] == "not_insect"]["filename"].tolist()

    ins_tracks = [(sx, sy) for fn in ins_fns
                  for sx, sy in [load_smooth(tracks_dir, fn, frame_height)] if sx is not None]
    ni_tracks  = [(sx, sy) for fn in ni_fns
                  for sx, sy in [load_smooth(tracks_dir, fn, frame_height)] if sx is not None]

    print(f"  Loaded         : {len(ins_tracks)} insect, {len(ni_tracks)} non-insect tracks")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    # Non-insect drawn first (background)
    ni_alpha = max(0.18, min(0.25, 20 / max(len(ni_tracks), 1)))
    for sx, sy in ni_tracks:
        ax.plot(sx, sy, color=RED, linewidth=1, alpha=ni_alpha, zorder=2)

    # Insect on top — slightly thicker, more opaque
    ins_alpha = max(0.40, min(0.85, 5 / max(len(ins_tracks), 1)))
    for sx, sy in ins_tracks:
        ax.plot(sx, sy, color=BLUE, linewidth=1.4, alpha=ins_alpha, zorder=3)

    # Start-point dots
    for sx, sy in ni_tracks:
        ax.plot(sx[0], sy[0], ".", ms=2.5, color=RED,  alpha=0.35, zorder=4)
    for sx, sy in ins_tracks:
        ax.plot(sx[0], sy[0], ".", ms=4.0, color=BLUE, alpha=0.70, zorder=5)

    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (pixels)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("y  (pixels)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    legend_handles = [
        plt.Line2D([0], [0], color=BLUE, linewidth=2.2, alpha=0.8,
                   label=f"Insect  (n = {len(ins_tracks)})"),
        plt.Line2D([0], [0], color=RED,  linewidth=2.2, alpha=0.8,
                   label=f"Non-insect  (n = {len(ni_tracks)})"),
        plt.Line2D([0], [0], marker=".", color="black", ms=6,
                   linestyle="none", label="Track start"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=LEGEND_SIZE,
              loc="lower right")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = out_dir / f"session_trajectories_rank{rank}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=180)
        print(f"  Saved → {out_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
