"""
Plot top positive and negative leave-one-group-out ablation effects.

Positive metric drop means the removed group was useful. Negative metric drop
means the model improved when the group was removed.
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


DEFAULT_INPUT = Path("data/results/six_domain_v1/ablation/group_loo_ablation.csv")
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 15
BAR_VALUE_SIZE = 15
SECTION_LABEL_SIZE = 13
DOMAIN_COLORS = {
    "Domain1": "#1f77b4",
    "Domain2": "#ff7f0e",
    "Domain3": "#2ca02c",
    "Domain4": "#d62728",
    "Domain5": "#9467bd",
    "Domain6": "#8c564b",
}
GROUP_LABELS = {
    "d1_duration": "Track support",
    "d1_speed": "Speed statistics",
    "d1_dynamic": "Dynamics",
    "d1_distributional": "Speed distribution",
    "d1_intensity": "Motion intensity",
    "d2_path_geometry": "Path geometry",
    "d2_spatial_spread": "Spatial spread",
    "d2_exploration": "Exploration",
    "d2_diffusion": "Diffusion",
    "d2_sinuosity": "Sinuosity",
    "d3_h_oscillation": "Horizontal oscillation",
    "d3_v_oscillation": "Vertical oscillation",
    "d3_cross_axis": "XY coherence",
    "d3_spectral": "Global spectral",
    "d3_residual": "Wingbeat residuals",
    "d3_hover_stab": "Hover control",
    "d4_directional": "Directionality",
    "d4_heading_occ": "Heading use",
    "d4_zigzag": "Zigzag & casting",
    "d4_looping": "Circling",
    "d4_bias": "Directional bias",
    "d5_turning": "Turning",
    "d5_saccade": "Saccades",
    "d5_fixation": "Fixation structure",
    "d5_maneuver": "Maneuverability",
    "d5_power_law": "Motor control",
    "d5_levy": "Levy statistics",
    "d5_aero": "Aerodynamics",
    "d6_hover": "Hover-burst",
    "d6_stopping": "Stops",
    "d6_landing": "Landing",
    "d6_takeoff": "Takeoff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot leave-one-group-out ablation effects")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--metric",
        choices=["f1", "pr_auc"],
        default="f1",
        help="Metric change to plot. For PR-AUC, the drop is computed from the FULL row.",
    )
    return parser.parse_args()


def prettify_group(name: str) -> str:
    if name in GROUP_LABELS:
        return GROUP_LABELS[name]
    parts = name.split("_")
    if parts and parts[0] in {"d1", "d2", "d3", "d4", "d5", "d6"}:
        parts = parts[1:]
    return " ".join(parts)


def metric_drop_column(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    if metric == "f1":
        if "f1_drop" not in df.columns:
            raise SystemExit("Input CSV must contain an f1_drop column")
        return df, "f1_drop", "F1 change"

    if "pr_auc_drop" not in df.columns:
        if "pr_auc" not in df.columns:
            raise SystemExit("Input CSV must contain pr_auc or pr_auc_drop for PR-AUC plotting")
        full = df[df["group"] == "FULL"]
        if full.empty:
            raise SystemExit("Input CSV must contain a FULL row to compute PR-AUC drop")
        baseline = float(full.iloc[0]["pr_auc"])
        df["pr_auc_drop"] = baseline - df["pr_auc"].astype(float)
    return df, "pr_auc_drop", "PR-AUC change"


def select_groups(df: pd.DataFrame, top_n: int, metric: str) -> tuple[pd.DataFrame, str, str]:
    df, drop_col, axis_label = metric_drop_column(df, metric)
    if "group" not in df.columns:
        raise SystemExit("Input CSV must contain a group column")

    if metric == "pr_auc":
        df = df[df["group"] != "FULL"].copy()
        highest = df.nlargest(top_n, drop_col).copy()
        least = df.nsmallest(top_n, drop_col).copy()
        highest["effect"] = "Highest contribution"
        least["effect"] = "Least contribution"
        selected = pd.concat([highest, least], ignore_index=True)
        selected = selected.drop_duplicates(subset=["group"], keep="first")
        if selected.empty:
            raise SystemExit(f"No {drop_col} rows found")
        selected["group_label"] = selected["group"].map(prettify_group)
        return selected.sort_values(drop_col, ascending=True).reset_index(drop=True), drop_col, axis_label

    positive = df[df[drop_col] > 0].nlargest(top_n, drop_col)
    negative = df[df[drop_col] < 0].nsmallest(top_n, drop_col)
    selected = pd.concat([positive, negative], ignore_index=True)
    if selected.empty:
        raise SystemExit(f"No positive or negative {drop_col} rows found")
    selected["effect"] = selected[drop_col].apply(
        lambda v: "Positive contribution" if v > 0 else "Negative contribution"
    )
    selected["group_label"] = selected["group"].map(prettify_group)
    return selected.sort_values(drop_col, ascending=True).reset_index(drop=True), drop_col, axis_label


def section_split(selected: pd.DataFrame) -> tuple[int, float] | None:
    effects = selected["effect"].tolist() if "effect" in selected.columns else []
    split_points = [
        idx for idx in range(1, len(effects))
        if effects[idx] != effects[idx - 1]
    ]
    if set(effects) == {"Least contribution", "Highest contribution"} and len(split_points) == 1:
        split_idx = split_points[0]
        return split_idx, split_idx - 0.5
    return None


def add_section_divider(ax: plt.Axes, selected: pd.DataFrame, show_labels: bool = False) -> None:
    split = section_split(selected)
    if split is None:
        return

    split_idx, boundary = split
    ax.axhline(
        boundary,
        color="0.15",
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        alpha=0.9,
        zorder=0,
    )
    if not show_labels:
        return

    label_transform = ax.get_yaxis_transform()
    ax.text(
        0.98, boundary - 0.35, f"{split_idx} least contributors",
        transform=label_transform, va="top", ha="right",
        fontsize=SECTION_LABEL_SIZE, color="0.25", fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    ax.text(
        0.98, boundary + 0.35, f"{len(selected) - split_idx} highest contributors",
        transform=label_transform, va="bottom", ha="right",
        fontsize=SECTION_LABEL_SIZE, color="0.25", fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )


def add_domain_legend(ax: plt.Axes, selected: pd.DataFrame) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=domain)
        for domain, color in DOMAIN_COLORS.items()
        if domain in set(selected["domain"])
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=LEGEND_SIZE)


def plot(selected: pd.DataFrame, output_png: Path, drop_col: str, axis_label: str) -> None:
    height = max(6.0, 0.46 * len(selected) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, height), dpi=180)

    colors = selected["domain"].map(DOMAIN_COLORS).fillna("#666666")
    ax.barh(selected["group_label"], selected[drop_col], color=colors)
    ax.axvline(0, color="0.2", linewidth=0.9)

    max_abs = abs(selected[drop_col]).max()
    pad = max(max_abs * 0.04, 0.0005)
    for y, value in enumerate(selected[drop_col]):
        if value < 0:
            ha = "left"
            x = pad * 0.7
            color = "black"
            weight = "normal"
        elif abs(value) > max_abs * 0.35:
            ha = "right"
            x = value - pad
            color = "white"
            weight = "bold"
        else:
            ha = "left"
            x = value + pad
            color = "black"
            weight = "normal"
        ax.text(
            x, y, f"{value:+.4f}",
            va="center", ha=ha, fontsize=BAR_VALUE_SIZE,
            color=color, fontweight=weight,
        )

    xmin = min(float(selected[drop_col].min()) - pad * 1.5, -pad * 2)
    xmax = max(float(selected[drop_col].max()) + pad * 2, pad * 3)
    ax.set_xlim(xmin, xmax)
    add_section_divider(ax, selected, show_labels=True)
    ax.set_xlabel(axis_label, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    add_domain_legend(ax, selected)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve() if args.output_png else \
        input_csv.with_name("group_loo_top_positive_negative.png")
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else \
        input_csv.with_name("group_loo_top_positive_negative.csv")

    df = pd.read_csv(input_csv)
    selected, drop_col, axis_label = select_groups(df, args.top_n, args.metric)
    selected.to_csv(output_csv, index=False)
    plot(selected, output_png, drop_col, axis_label)

    print(f"Selected ablation rows -> {output_csv}")
    print(f"Ablation plot          -> {output_png}")


if __name__ == "__main__":
    main()
